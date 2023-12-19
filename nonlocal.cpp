#include "nonlocal.hpp"
#include "filter.hpp"
#include "utilities.hpp"
#include <Teuchos_ENull.hpp>
#include <cstdlib>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <thread>

Nonlocal::Nonlocal(Teuchos::RCP<Teuchos::ParameterList> &params)
    : params_(params), mpi_communicator_(MPI_COMM_WORLD),
      triangulation_(
          MPI_COMM_WORLD,
          dealii::Triangulation<DIM>::limit_level_difference_at_vertices,
          dealii::parallel::distributed::Triangulation<
              DIM>::construct_multigrid_hierarchy),
      dof_handler_(triangulation_), density_dof_handler_(triangulation_),
      fe_(dealii::FE_Q<DIM>(params->get<int>("finite element order")), DIM),
      density_fe_(0), stress_calculator_(1., 0.3),
      pcout(std::cout,
            dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      timer_(mpi_communicator_, pcout, dealii::TimerOutput::summary,
             dealii::TimerOutput::wall_times),
      ParOptProblem(MPI_COMM_WORLD) {

  this->incref();

  statistics_.set_auto_fill_mode(true);
  statistics_postprocess_.set_auto_fill_mode(true);

  output_dir_ = params_->get<std::string>("output directory name");
  pcout << "Output directory name: " << output_dir_ << std::endl;
  resume_ = (params_->get<bool>("resume", false) ||
             params_->get<std::string>("mode") == "postprocess");

  std::string mesh_filename = params->get<std::string>("mesh file name");
  dealii::GridIn<DIM> grid_in(triangulation_);
  grid_in.read(mesh_filename, dealii::GridIn<DIM>::Format::msh);

  dealii::GridTools::scale(params_->get<double>("mesh scale", 1.),
                           triangulation_);

  triangulation_.refine_global(
      params_->get<int>("global mesh refinement steps"));

  optimization_iteration_ = 0;

  setup_system();
}

void Nonlocal::setup_system() {

  dealii::TimerOutput::Scope t(timer_, "setup system");

  mg_matrices_.clear_elements();

  dof_handler_.distribute_dofs(fe_);
  dof_handler_.distribute_mg_dofs();

  density_dof_handler_.distribute_dofs(density_fe_);
  density_dof_handler_.distribute_mg_dofs();

  pcout << "No. of displacement DoFs: " << dof_handler_.n_dofs() << std::endl;
  pcout << "No. of density DoFs: " << density_dof_handler_.n_dofs()
        << std::endl;

  dealii::IndexSet relevant_set;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_, relevant_set);

  constraints_.clear();
  constraints_.reinit(relevant_set);
  boundary_displacements_ =
      params_->get<Teuchos::Array<double>>("boundary displacements");
  setup_constraints(constraints_, dof_handler_, boundary_displacements_, 1.);
  constraints_.close();

  {

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_,
                                                    locally_relevant_dofs);

    system_rhs_.reinit(dof_handler_.locally_owned_dofs(), locally_relevant_dofs,
                       mpi_communicator_);
    solution_.reinit(dof_handler_.locally_owned_dofs(), locally_relevant_dofs,
                     mpi_communicator_);
    lagrange_multiplier_vector_.reinit(dof_handler_.locally_owned_dofs(),
                                       locally_relevant_dofs,
                                       mpi_communicator_);
    solution_ = 0.;
    lagrange_multiplier_vector_ = 0.;
  }

  const auto density_dofs = density_dof_handler_.locally_owned_dofs();
  design_.reinit(density_dofs, mpi_communicator_);
  material_.reinit(density_dofs, mpi_communicator_);

  design_old_.reinit(design_);
  is_design_.reinit(design_);

  is_design_ = 1.;

  const auto solid_domain_ids = params_->get<Teuchos::Array<int>>(
      "solid domain ids", Teuchos::Array<int>());
  for (const auto &cell : density_dof_handler_.active_cell_iterators()) {
    if (cell->is_locally_owned())
      if (std::find(solid_domain_ids.begin(), solid_domain_ids.end(),
                    cell->material_id()) != solid_domain_ids.end()) {
        is_design_[cell->dof_index(0)] = 0.;
      }
  }
  is_design_.compress(dealii::VectorOperation::insert);

  double cell_measure;
  for (const auto &cell : dof_handler_.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      cell_measure = cell->measure();
      break;
    }
  }
  if (params_->isParameter("target volume")) {
    target_volume_ = params_->get<double>("target volume");
    vol_frac_ = (target_volume_ -
                 (is_design_.size() - is_design_.l1_norm()) * cell_measure) /
                (is_design_.l1_norm() * cell_measure);
    params_->set<double>("volume fraction", vol_frac_);
  } else {
    vol_frac_ = params_->get<double>("volume fraction");
    target_volume_ = vol_frac_ * is_design_.l1_norm() * cell_measure +
                     (is_design_.size() - is_design_.l1_norm()) * cell_measure;
  }
  design_ = vol_frac_;
  pcout << "Total volume: " << is_design_.size() * cell_measure << std::endl;
  pcout << "Torget volume: " << target_volume_ << std::endl;
  pcout << "Volume fraction: " << vol_frac_ << std::endl;
  pcout << "Non-design volume: "
        << (is_design_.size() - is_design_.l1_norm()) * cell_measure
        << std::endl;

  sensitivity_.reinit(design_);
  sensitivity_ = 0.;

  {
    stiffness_operator_ = Teuchos::rcp(new StiffnessOperator<double>());
    stiffness_operator_->clear();

    std::vector<const dealii::DoFHandler<DIM> *> dofs;
    std::vector<const dealii::AffineConstraints<double> *> cons;
    dofs.push_back(&dof_handler_);
    dofs.push_back(&density_dof_handler_);
    cons.push_back(&constraints_);
    empty_constraints_.reinit(density_dof_handler_.locally_owned_dofs());
    empty_constraints_.close();
    cons.push_back(&empty_constraints_);

    typename dealii::MatrixFree<DIM, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = dealii::MatrixFree<
        DIM, double>::AdditionalData::TasksParallelScheme::partition_partition;
    additional_data.mapping_update_flags =
        (dealii::update_gradients | dealii::update_JxW_values |
         dealii::update_quadrature_points | dealii::update_values);

    std::shared_ptr<dealii::MatrixFree<DIM, double>> system_mf_storage(
        new dealii::MatrixFree<DIM, double>());
    system_mf_storage->reinit(mapping_, dofs, cons,
                              std::vector<dealii::QGauss<1>>{2, 1},
                              additional_data);
    stiffness_operator_->initialize(system_mf_storage,
                                    std::vector<unsigned int>{0});

    stiffness_operator_->initialize_dof_vector(solution_, 0);
    stiffness_operator_->initialize_dof_vector(system_rhs_, 0);
    stiffness_operator_->initialize_dof_vector(elastic_modulus_, 1);

    //////////////////////////////////////////////////////////////////////////

    const unsigned int nlevels = triangulation_.n_global_levels();
    mg_matrices_.resize(0, nlevels - 1);
    pcout << "MG levels: " << nlevels << std::endl;

    mg_transfer_.build(density_dof_handler_);
    mg_densities_.resize(0, nlevels - 1);

    std::set<dealii::types::boundary_id> dirichlet_boundary_ids;
    for (int i = 0; i < boundary_displacements_.size(); i += 3) {
      dirichlet_boundary_ids.insert(int(round(boundary_displacements_[i])));
    }

    mg_constrained_dofs.initialize(dof_handler_);

    // FIXME: correct for inhomogeneous constraints
    for (int i = 0; i < boundary_displacements_.size(); i += 3) {
      const int boundary_id = round(boundary_displacements_[i]);
      const int mask = round(boundary_displacements_[i + 1]);
      const double value = boundary_displacements_[i + 2];
      if (DIM == 2)
        mg_constrained_dofs.make_zero_boundary_constraints(
            dof_handler_, std::set<dealii::types::boundary_id>{boundary_id},
            std::vector<bool>{mask / 10 % 10, mask % 10});
      else
        mg_constrained_dofs.make_zero_boundary_constraints(
            dof_handler_, std::set<dealii::types::boundary_id>{boundary_id},
            std::vector<bool>{mask / 100 % 10, mask / 10 % 10, mask % 10});
    }

    for (unsigned int level = 0; level < nlevels; ++level) {
      // TODO: use this for newer versions of dealii
      // dealii::AffineConstraints<double> level_constraints(
      //     dof_handler_.locally_owned_mg_dofs(level),
      //     dealii::DoFTools::extract_locally_relevant_level_dofs(dof_handler_,
      //                                                           level));
      dealii::AffineConstraints<double> level_constraints(
          dealii::DoFTools::extract_locally_relevant_level_dofs(dof_handler_,
                                                                level));
      level_constraints.add_lines(
          mg_constrained_dofs.get_boundary_indices(level));
      level_constraints.close();
      std::vector<const dealii::AffineConstraints<double> *> mg_level_cons;
      mg_level_cons.push_back(&level_constraints);
      dealii::AffineConstraints<double> level_constraints_density(
          density_dof_handler_.locally_owned_mg_dofs(level));
      level_constraints_density.close();
      mg_level_cons.push_back(&level_constraints_density);

      typename dealii::MatrixFree<DIM, float>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
          dealii::MatrixFree<DIM, float>::AdditionalData::partition_partition;
      additional_data.mapping_update_flags =
          (dealii::update_values | dealii::update_gradients |
           dealii::update_quadrature_points | dealii::update_JxW_values);
      additional_data.mg_level = level;
      std::shared_ptr<dealii::MatrixFree<DIM, float>> mg_mf_storage_level(
          new dealii::MatrixFree<DIM, float>());
      mg_mf_storage_level->reinit(mapping_, dofs, mg_level_cons,
                                  std::vector<dealii::QGauss<1>>{2, 1},
                                  additional_data);
      mg_matrices_[level].initialize(mg_mf_storage_level, mg_constrained_dofs,
                                     level, std::vector<unsigned int>{0});

      mg_matrices_[level].initialize_dof_vector(mg_densities_[level], 1);
    }
  }

  assemble_rhs();

  {
    VectorType tmp(system_rhs_);
    for (const auto i : tmp.locally_owned_elements())
      tmp[i] = std::abs(tmp[i]) > 1.e-8 ? 1. : 0.;
    mass_matrix_.reinit(tmp);
  }

  const std::string model_name = params_->get<std::string>("model_name");

  pcout << "Model: " << model_name << std::endl;

  if (model_name == "nonlocal")
    nonlocal_model_ = Teuchos::rcp(
        new NonlocalModel(params_, density_dof_handler_, is_design_, pcout));
  else if (model_name == "simp")
    nonlocal_model_ = Teuchos::rcp(
        new StandardSIMP(params_, density_dof_handler_, is_design_, pcout));
  else if (model_name == "simp_heaviside")
    nonlocal_model_ = Teuchos::rcp(
        new SIMPHeaviside(params_, density_dof_handler_, is_design_, pcout));
  else if (model_name == "nonlocal_heaviside")
    nonlocal_model_ = Teuchos::rcp(new NonlocalModelHeaviside(
        params_, density_dof_handler_, is_design_, pcout));
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "Model " << model_name << " not implemented");
  }

  design_lb.reinit(design_);
  design_ub.reinit(design_);
  design_lb = 0.;
  design_ub = 1.;

  for (auto i : is_design_.locally_owned_elements()) {
    if (is_design_[i] < 0.5) {
      design_lb[i] = 1. - 1.e-9;
      design_[i] = 1.;
    }
  }

  design_lb.compress(dealii::VectorOperation::insert);
  design_.compress(dealii::VectorOperation::insert);

  perturb_vector(design_, design_lb, design_ub,
                 params_->get<double>("random perturbation scale"));

  design_old_ = design_;

  if (resume_) {

    const std::string density_file = params_->get<std::string>("density file");
    read_material_density_from_file(density_dof_handler_, design_, material_,
                                    density_file, mpi_communicator_);
    if (params_->get<bool>("use material for initial design"))
      design_ = material_;
  }
}

void Nonlocal::assemble_rhs() {

  system_rhs_ = 0.;

  dealii::FEValuesExtractors::Vector displacement(0);

  dealii::QGauss<DIM - 1> face_quadrature_formula(fe_.degree + 1);
  dealii::FEFaceValues<DIM> fe_face_values(fe_, face_quadrature_formula,
                                           dealii::update_values |
                                               dealii::update_JxW_values |
                                               dealii::update_normal_vectors);
  dealii::Vector<double> cell_vector(fe_face_values.dofs_per_cell);
  std::vector<dealii::types::global_dof_index> local_dof_indices(
      fe_face_values.dofs_per_cell);

  const auto boundary_tractions = params_->get<Teuchos::Array<double>>(
      "boundary tractions", Teuchos::Array<double>());
  for (int n = 0; n < boundary_tractions.size(); n += 3) {

    const int boundary_id = round(boundary_tractions[n]);
    const int mask = round(boundary_tractions[n + 1]);
    const double value = boundary_tractions[n + 2];

    dealii::Tensor<1, DIM> traction;
    if (DIM == 2) {
      traction[0] = (mask / 10 % 10) * value;
      traction[1] = (mask % 10) * value;
    } else {
      traction[0] = (mask / 100 % 10) * value;
      traction[1] = (mask / 10 % 10) * value;
      traction[2] = (mask % 10) * value;
    }

    for (const auto &cell : dof_handler_.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;

      cell_vector = 0.;

      if (cell->at_boundary()) {
        for (const auto f : dealii::GeometryInfo<DIM>::face_indices()) {
          if (cell->face(f)->at_boundary()) {
            if (cell->face(f)->boundary_id() == boundary_id) {
              fe_face_values.reinit(cell, f);
              for (const auto q : fe_face_values.quadrature_point_indices()) {
                for (const auto i : fe_face_values.dof_indices()) {
                  cell_vector[i] += traction *
                                    fe_face_values[displacement].value(i, q) *
                                    fe_face_values.JxW(q);
                }
              }
            }
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);

      constraints_.distribute_local_to_global(cell_vector, local_dof_indices,
                                              system_rhs_);
    }
  }

  const auto boundary_pressure = params_->get<Teuchos::Array<double>>(
      "boundary pressure", Teuchos::Array<double>());
  for (int n = 0; n < boundary_pressure.size(); n += 2) {

    const int boundary_id = round(boundary_pressure[n]);
    const double value = boundary_pressure[n + 1];

    for (const auto &cell : dof_handler_.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;

      cell_vector = 0.;

      if (cell->at_boundary()) {
        for (const auto f : dealii::GeometryInfo<DIM>::face_indices()) {
          if (cell->face(f)->at_boundary()) {
            if (cell->face(f)->boundary_id() == boundary_id) {
              fe_face_values.reinit(cell, f);
              for (const auto q : fe_face_values.quadrature_point_indices()) {
                for (const auto i : fe_face_values.dof_indices()) {
                  cell_vector[i] -= value * fe_face_values.normal_vector(q) *
                                    fe_face_values[displacement].value(i, q) *
                                    fe_face_values.JxW(q);
                }
              }
            }
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);

      constraints_.distribute_local_to_global(cell_vector, local_dof_indices,
                                              system_rhs_);
    }
  }
  system_rhs_.compress(dealii::VectorOperation::add);
}

void Nonlocal::assemble_sensitivity_vector(LA::MPI::Vector &sensitivity) {

  dealii::TimerOutput::Scope t(timer_, "assemble sensitivity vector");

  solution_.update_ghost_values();
  lagrange_multiplier_vector_.update_ghost_values();

  dealii::Vector<double> local_u(fe_.dofs_per_cell);
  dealii::Vector<double> local_lambd(fe_.dofs_per_cell);
  dealii::Vector<double> local_Ku(fe_.dofs_per_cell);

  dealii::FEValuesExtractors::Vector displacement(0);

  dealii::QGauss<DIM> quadrature_formula(fe_.degree + 1);
  dealii::FEValues<DIM> fe_values(fe_, quadrature_formula,
                                  dealii::update_gradients |
                                      dealii::update_JxW_values);

  dealii::FullMatrix<double> cell_matrix(fe_values.dofs_per_cell,
                                         fe_values.dofs_per_cell);

  for (const auto &cell : dof_handler_.active_cell_iterators()) {
    if (cell->is_locally_owned()) {

      cell->get_dof_values(solution_, local_u);
      cell->get_dof_values(lagrange_multiplier_vector_, local_lambd);

      typename dealii::DoFHandler<DIM>::active_cell_iterator cell_density =
          cell->as_dof_handler_iterator(density_dof_handler_);

      cell_matrix = 0.;

      fe_values.reinit(cell);

      dealii::SymmetricTensor<2, DIM> shape_stress;
      for (const auto q : fe_values.quadrature_point_indices()) {
        for (const auto i : fe_values.dof_indices()) {
          stress_calculator_.calculate(
              shape_stress, fe_values[displacement].symmetric_gradient(i, q));
          for (const auto j : fe_values.dof_indices()) {

            cell_matrix(i, j) +=
                fe_values[displacement].symmetric_gradient(j, q) *
                shape_stress * fe_values.JxW(q);
          }
        }
      }

      cell_matrix.vmult(local_Ku, local_u);
      sensitivity[cell_density->dof_index(0)] =
          -(Emax - Emin) * (local_Ku * local_lambd);
    }
  }

  solution_.zero_out_ghost_values();
  lagrange_multiplier_vector_.zero_out_ghost_values();
}

void Nonlocal::solve_matrix_free(VectorType &solution,
                                 const LA::MPI::Vector &elastic_modulus,
                                 const VectorType &rhs) {
  dealii::TimerOutput::Scope t(timer_, "solve matrix free");

  const unsigned int nlevels = triangulation_.n_global_levels();

  {
    elastic_modulus_.reinit(density_dof_handler_.locally_owned_dofs(),
                            mpi_communicator_);
    std::copy(elastic_modulus.begin(), elastic_modulus.end(),
              elastic_modulus_.begin());
    stiffness_operator_->set_elastic_modulus(elastic_modulus_);
    mg_transfer_.interpolate_to_mg(density_dof_handler_, mg_densities_,
                                   elastic_modulus_);
    for (unsigned int level = 0; level < nlevels; ++level) {
      mg_matrices_[level].set_elastic_modulus(mg_densities_[level]);
    }
  }

  dealii::MGTransferMatrixFree<DIM, float> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler_);

  using SmootherType = dealii::PreconditionChebyshev<
      LevelMatrixType, dealii::LinearAlgebra::distributed::Vector<float>>;
  dealii::mg::SmootherRelaxation<
      SmootherType, dealii::LinearAlgebra::distributed::Vector<float>>
      mg_smoother;
  dealii::MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, triangulation_.n_global_levels() - 1);
  for (unsigned int level = 0; level < triangulation_.n_global_levels();
       ++level) {
    if (level > 0) {
      smoother_data[level].smoothing_range = 15.;
      smoother_data[level].degree = 4;
      smoother_data[level].eig_cg_n_iterations = 10;
    } else {
      smoother_data[0].smoothing_range = 1e-3;
      smoother_data[0].degree = dealii::numbers::invalid_unsigned_int;
      smoother_data[0].eig_cg_n_iterations = mg_matrices_[0].m();
    }

    mg_matrices_[level].compute_diagonal();
    smoother_data[level].preconditioner =
        mg_matrices_[level].get_matrix_diagonal_inverse();
  }
  mg_smoother.initialize(mg_matrices_, smoother_data);

  dealii::MGCoarseGridApplySmoother<
      dealii::LinearAlgebra::distributed::Vector<float>>
      mg_coarse;
  mg_coarse.initialize(mg_smoother);

  dealii::mg::Matrix<dealii::LinearAlgebra::distributed::Vector<float>>
      mg_matrix(mg_matrices_);

  dealii::MGLevelObject<
      dealii::MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
      mg_interface_matrices;
  mg_interface_matrices.resize(0, triangulation_.n_global_levels() - 1);
  for (unsigned int level = 0; level < triangulation_.n_global_levels();
       ++level) {
    mg_interface_matrices[level].initialize(mg_matrices_[level]);
  }
  dealii::mg::Matrix<dealii::LinearAlgebra::distributed::Vector<float>>
      mg_interface(mg_interface_matrices);

  dealii::Multigrid<dealii::LinearAlgebra::distributed::Vector<float>> mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
  mg.set_edge_matrices(mg_interface, mg_interface);

  dealii::PreconditionMG<DIM, dealii::LinearAlgebra::distributed::Vector<float>,
                         dealii::MGTransferMatrixFree<DIM, float>>
      preconditioner(dof_handler_, mg, mg_transfer);

  dealii::PreconditionJacobi<StiffnessOperator<double>> preconditioner_jacobi;
  preconditioner_jacobi.initialize(*stiffness_operator_);
  stiffness_operator_->compute_diagonal();

  constraints_.set_zero(solution);

  dealii::SolverControl solver_control(
      100, params_->get<double>("linear solver tolerance") * rhs.l2_norm());
  dealii::SolverCG<VectorType> solver(solver_control);
  try {
    solver.solve(*stiffness_operator_, solution, rhs, preconditioner);
  } catch (...) {
    pcout << "Solver failed to converge!" << std::endl;
  }
  constraints_.distribute(solution);
}

void Nonlocal::checkpoint(const std::string &filename_prefix,
                          const LA::MPI::Vector &design,
                          const LA::MPI::Vector &material) {

  dealii::TimerOutput::Scope t(timer_, "checkpoint");

  dealii::Table<2, double> table;
  table.reinit(density_dof_handler_.n_locally_owned_dofs(), DIM + 2);
  unsigned int row = 0;
  for (const auto &cell : density_dof_handler_.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      const auto &c = cell->center();
      for (int d = 0; d < DIM; ++d)
        table(row, d) = c[d];
      table(row, DIM) = design[cell->dof_index(0)];
      table(row, DIM + 1) = material[cell->dof_index(0)];
      ++row;
    }
  }
  const auto table_global =
      dealii::Utilities::MPI::gather(mpi_communicator_, table);

  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {
    std::ofstream out(output_dir_ + "/" + filename_prefix + ".dat");
    out << "# x y (z) design material" << std::endl;
    for (const auto &a : table_global) {
      for (unsigned int i = 0; i < a.n_rows(); ++i) {
        for (unsigned int j = 0; j < a.n_cols(); ++j)
          out << a(i, j) << " ";
        out << std::endl;
      }
    }
  }
}

void Nonlocal::output_results(
    const std::string &prefix, const VectorType &solution,
    const std::map<std::string, Teuchos::RCP<LA::MPI::Vector>> &density_vectors,
    const unsigned int cycle) {

  dealii::TimerOutput::Scope t(timer_, "output results");

  std::vector<std::string> solution_names(DIM, "u");
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
      interpretation(
          DIM,
          dealii::DataComponentInterpretation::component_is_part_of_vector);
  dealii::DataOut<DIM> data_out;

  solution.update_ghost_values();

  data_out.add_data_vector(dof_handler_, solution, solution_names,
                           interpretation);

  for (const auto &a : density_vectors)
    data_out.add_data_vector(density_dof_handler_, *a.second, a.first);

  // subdomain ids
  dealii::Vector<float> subdomain(triangulation_.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation_.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  dealii::DataOutBase::VtkFlags flags;
  flags.compression_level = dealii::DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(flags);

  const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(
      output_dir_ + "/", prefix, cycle, mpi_communicator_);

  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {
    times_and_names_.emplace_back(cycle, pvtu_filename);
    std::ofstream pvd_output(output_dir_ + "/" + prefix + ".pvd");
    dealii::DataOutBase::write_pvd_record(pvd_output, times_and_names_);
  }

  solution.zero_out_ghost_values();
}

//! Get the variables/bounds
void Nonlocal::getVarsAndBounds(ParOptVec *xvec, ParOptVec *lbvec,
                                ParOptVec *ubvec) {
  ParOptScalar *x, *lb, *ub;
  xvec->getArray(&x);
  lbvec->getArray(&lb);
  ubvec->getArray(&ub);

  assert(nvars == density_dof_handler_.n_locally_owned_dofs());

  // Set the design variable bounds
  unsigned int i = 0;
  for (const auto global_index : design_.locally_owned_elements()) {
    x[i] = design_[global_index];
    lb[i] = design_lb[global_index];
    ub[i] = design_ub[global_index];
    ++i;
  }
}

ParOptQuasiDefMat *Nonlocal::createQuasiDefMat() {
  int nwblock = 0;
  return new ParOptQuasiDefBlockMat(this, nwblock);
}

void Nonlocal::write_current_visualization() {
  std::map<std::string, Teuchos::RCP<LA::MPI::Vector>> density_vectors;

  nonlocal_model_->set_design(design_);
  density_vectors["material"] = nonlocal_model_->get_material();
  density_vectors["design"] = nonlocal_model_->get_design();
  density_vectors["elastic_modulus"] = nonlocal_model_->get_elastic_modulus();
  density_vectors["filtered_density_2"] =
      nonlocal_model_->get_filtered_density_2();
  density_vectors["is_design"] = Teuchos::rcpFromRef(is_design_);
  density_vectors["sensitivity"] = Teuchos::rcpFromRef(sensitivity_);
  density_vectors["skeleton"] = nonlocal_model_->get_skeleton();

  output_results("solution", solution_, density_vectors,
                 optimization_iteration_);
}

int Nonlocal::evalObjCon(ParOptVec *xvec, ParOptScalar *fobj,
                         ParOptScalar *cons) {

  dealii::TimerOutput::Scope t(timer_, "eval obj/con");

  ParOptScalar *x;
  xvec->getArray(&x);

  LA::MPI::Vector design(design_);

  std::copy(x, x + design.locally_owned_size(), design.begin());

  design_ = design;

  nonlocal_model_->set_design(design);
  const auto elastic_modulus = nonlocal_model_->get_elastic_modulus();

  solve_matrix_free(solution_, *elastic_modulus, system_rhs_);

  auto c = nonlocal_model_->get_constraint_value();

  const double compliance = solution_ * system_rhs_;
  const double volume_fraction = nonlocal_model_->get_volume_fraction();

  if (set_objective_scale_) {
    obj_scale_ = 10. / compliance;
    set_objective_scale_ = false;
  }

  *fobj = compliance * obj_scale_;
  cons[0] = -volume_fraction + vol_frac_;

  return 0;
}

void Nonlocal::write_statistics() {

  dealii::TimerOutput::Scope t(timer_, "write_statistics");

  const double compliance = solution_ * system_rhs_;

  statistics_.add_value("compliance", compliance);
  statistics_.set_precision("compliance", 6);
  statistics_.set_scientific("compliance", 6);

  const double volume = nonlocal_model_->get_volume_fraction();
  statistics_.add_value("volume", volume);
  statistics_.set_precision("volume", 6);
  statistics_.set_scientific("volume", 6);

  const double total_volume = nonlocal_model_->get_total_volume();
  statistics_.add_value("total_volume", total_volume);
  statistics_.set_precision("total_volume", 6);
  statistics_.set_scientific("total_volume", 6);

  const double beta = nonlocal_model_->get_continuation_parameter();
  statistics_.add_value("beta", beta);
  statistics_.set_precision("beta", 6);
  statistics_.set_scientific("beta", 6);

  const auto &design = *nonlocal_model_->get_design();
  design_old_ -= design;
  const double change = design_old_.linfty_norm();
  design_old_ = design;
  statistics_.add_value("change", change);
  statistics_.set_precision("change", 6);
  statistics_.set_scientific("change", 6);

  const double grayness = calculate_grayness(*nonlocal_model_->get_material());
  statistics_.add_value("grayness", grayness);
  statistics_.set_precision("grayness", 6);
  statistics_.set_scientific("grayness", 6);

  const double duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - wall_clock_time_)
          .count();

  wall_clock_time_ = std::chrono::steady_clock::now();
  statistics_.add_value("duration [ms]", duration);
  statistics_.set_precision("duration [ms]", 6);
  statistics_.set_scientific("duration [ms]", 6);

  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {
    std::ofstream stat_file((output_dir_ + "/statistics.dat").c_str());
    statistics_.write_text(
        stat_file,
        dealii::TableHandler::simple_table_with_separate_column_description);
    stat_file.close();
  }

  total_time_ += duration;

  {
    // Print the output to the screen
    pcout << " Iteration: " << optimization_iteration_;
    pcout << " Objective: " << compliance;
    if (nonlocal_model_->needs_continuation())
      pcout << " Beta: " << beta;
    pcout << " Volume frac: " << volume;
    pcout << " Total volume: " << total_volume;
    pcout << " Change: " << change;
    pcout << " Grayness: " << grayness;
    pcout << " Duration: " << duration / 1000. << " s";
    pcout << " Total time: " << total_time_ / 1000. << " s";
    pcout << std::endl;
  }
}

int Nonlocal::evalObjConGradient(ParOptVec *xvec, ParOptVec *gvec,
                                 ParOptVec **Ac) {
  dealii::TimerOutput::Scope t(timer_, "eval obj/con grad");
  ++optimization_iteration_;

  ParOptScalar *x, *g, *c, *c2;
  xvec->getArray(&x);
  gvec->getArray(&g);

  Ac[0]->getArray(&c);

  LA::MPI::Vector design(design_);

  std::copy(x, x + design.locally_owned_size(), design.begin());

  design_ = design;

  LA::MPI::Vector dc(design);
  LA::MPI::Vector dv(design);
  LA::MPI::Vector boundary_constraint_grad(design);

  nonlocal_model_->set_design(design);

  lagrange_multiplier_vector_ = solution_;
  assemble_sensitivity_vector(sensitivity_);
  nonlocal_model_->filter_sensitivity(sensitivity_);

  dc = sensitivity_;

  nonlocal_model_->get_volume_fraction_gradient(dv);

  dc *= obj_scale_;
  std::copy(dc.begin(), dc.end(), g);
  dv *= -1.;
  std::copy(dv.begin(), dv.end(), c);

  if (write_results_) {
    if (optimization_iteration_ % params_->get<int>("output frequency") == 0)
      write_current_visualization();
    if (optimization_iteration_ % params_->get<int>("checkpoint frequency") ==
        0)
      checkpoint("design", *nonlocal_model_->get_design(),
                 *nonlocal_model_->get_material());
    write_statistics();
  }

  return 0;
}

void Nonlocal::optimize_paropt(const unsigned int max_iters) {

  setProblemSizes(density_dof_handler_.n_locally_owned_dofs(), 1, 0);

  // Create the options class, and create default values
  ParOptOptions *options = new ParOptOptions();
  ParOptOptimizer::addDefaultOptions(options);

  options->setOption("algorithm", "mma");
  options->setOption("max_major_iters", 100); // default value is 5000
  options->setOption("mma_max_iterations", (int)max_iters);
  options->setOption("abs_res_tol", 1.e-15);
  options->setOption("starting_point_strategy", "affine_step");
  options->setOption("barrier_strategy", "mehrotra_predictor_corrector");
  options->setOption("use_line_search", false);
  options->setOption("output_level", 1);
  options->setOption("mma_output_file", (output_dir_ + "/paropt.mma").c_str());
  options->setOption("output_file", (output_dir_ + "/paropt.out").c_str());
  if (nonlocal_model_->needs_continuation() &&
      params_->get<bool>("adapt move limit", false))
    options->setOption(
        "mma_move_limit",
        0.2 / (1. + std::sqrt(nonlocal_model_->get_continuation_parameter())));

  ParOptOptimizer *opt = new ParOptOptimizer(this, options);
  opt->incref();
  opt->optimize();
  opt->decref();
}

void Nonlocal::check_gradients() {
  write_results_ = false;
  set_objective_scale_ = false;

  setProblemSizes(density_dof_handler_.n_locally_owned_dofs(), 1, 0);

  LA::MPI::Vector x(design_);
  x = design_;
  for (int i = 0; i < 10; ++i) {
    this->checkGradients(0.1 / pow(10., i));
    design_ = x;
    pcout << std::endl;
  }
}

void Nonlocal::scale_design_to_satisfy_constraint(
    LA::MPI::Vector &design) const {

  double l1 = 0.;
  double l2 = 1.;

  LA::MPI::Vector x_new(design);

  while (std::abs(l2 - l1) / (l1 + l2) > 1.e-5) {

    const double scale = 0.5 * (l1 + l2);

    for (const auto i : design.locally_owned_elements()) {
      if (is_design_[i] > 0.5) {
        x_new[i] = design[i] * scale;
      }
    }
    nonlocal_model_->set_design(x_new);
    const double val = nonlocal_model_->get_constraint_value();
    if (val > 0.)
      l2 = scale;
    else
      l1 = scale;
  }

  design = x_new;
}

void Nonlocal::optimize() {

  solution_ = 0.;

  unsigned int max_iters = params_->get<int>("max iterations");

  if (nonlocal_model_->needs_continuation()) {
    const auto betas = params_->get<Teuchos::Array<double>>("beta");
    const auto iters_per_beta = params_->get<int>("iterations per beta");

    for (int beta_iter = 0; beta_iter < betas.size(); ++beta_iter) {
      nonlocal_model_->set_continuation_parameter(betas[beta_iter]);

      const unsigned int niters =
          beta_iter == betas.size() - 1 ? max_iters : iters_per_beta;

      write_current_visualization();
      optimize_paropt(niters);
    }
  } else {
    write_current_visualization();
    optimize_paropt(max_iters);
  }

  write_current_visualization();
  checkpoint("design", *nonlocal_model_->get_design(),
             *nonlocal_model_->get_material());
}

void Nonlocal::postprocess(const LA::MPI::Vector &gray_material) {

  LA::MPI::Vector material(gray_material);
  material = gray_material;
  remove_grayness(material, is_design_, vol_frac_);

  LA::MPI::Vector elastic_modulus(material);
  nonlocal_model_->apply_raw_material_filter(elastic_modulus, material);

  VectorType solution(solution_);

  solution = 0.;
  solve_matrix_free(solution, elastic_modulus, system_rhs_);
  const double nonlocal_compliance = solution * system_rhs_;

  solution = 0.;
  solve_matrix_free(solution, material, system_rhs_);
  const double simp_compliance = solution * system_rhs_;

  std::map<std::string, Teuchos::RCP<LA::MPI::Vector>> density_vectors;
  density_vectors["material"] = Teuchos::rcpFromRef(material);
  density_vectors["elastic_modulus"] = Teuchos::rcpFromRef(elastic_modulus);
  output_results("postprocess", solution, density_vectors, 0);

  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {
    std::ofstream file(output_dir_ + "/" + "postprocess_compliance.dat");
    file << "# nonlocal_compliance    simp_compliance" << std::endl;
    file << nonlocal_compliance << "   " << simp_compliance << std::endl;
  }
  checkpoint("bw_design", material, material);
}

void Nonlocal::run() {

  {
    const unsigned int n_vect_doubles = dealii::VectorizedArray<double>::size();
    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

    pcout << "Vectorization over " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << dealii::Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
  }

  std::string mode = params_->get<std::string>("mode");

  bool do_postprocess = false;

  LA::MPI::Vector design(design_);
  for (const auto i : design_.locally_owned_elements())
    design[i] = is_design_[i] > 0.5 ? design_ub[i] : design_[i];
  nonlocal_model_->set_design(design);
  const auto elastic_modulus = nonlocal_model_->get_elastic_modulus();
  solution_ = 0.;
  solve_matrix_free(solution_, *elastic_modulus, system_rhs_);
  compliance_lb_ = solution_ * system_rhs_;

  if (mode == "optimization") {
    optimize();
    nonlocal_model_->set_design(design_);
    material_ = *nonlocal_model_->get_material();
    postprocess(material_);
  } else if (mode == "postprocess") {
    if (nonlocal_model_->needs_continuation()) {
      const auto betas = params_->get<Teuchos::Array<double>>("beta");
      nonlocal_model_->set_continuation_parameter(betas.back());
    }
    postprocess(material_);
  } else if (mode == "gradient_check") {
    check_gradients();
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "Unknown mode: " << mode);
  }
}
