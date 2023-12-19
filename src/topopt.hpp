#pragma once

#include <algorithm>
#include <chrono>
#include <deal.II/base/geometry_info.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/numerics/vector_tools_boundary.h>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include <deal.II/base/aligned_vector.h>

#include "Teuchos_StandardCatchMacros.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/base/timer.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include "filter.hpp"
#include "global.hpp"

#include "ParOptOptimizer.h"
#include "ParOptProblem.h"

#include <deal.II/numerics/rtree.h>

class Stress {

public:
  Stress(const double E = 1., const double nu = 0.3) : E_(E), nu_{nu} {
    lame_lambda_ = E_ * nu_ / ((1. + nu_) * (1. - 2. * nu_));
    lame_mu_ = E_ / (2. * (1. + nu_));
  }

  template <typename number>
  void convert_strain_to_stress(
      dealii::SymmetricTensor<2, DIM, number> &value) const {
    const auto tmp = lame_lambda_ * dealii::trace(value);
    value *= 2. * lame_mu_;
    for (int i = 0; i < DIM; ++i)
      value[i][i] += tmp;
  }

  void calculate(dealii::SymmetricTensor<2, DIM> &stress,
                 const dealii::SymmetricTensor<2, DIM> &strain) const {

    stress = 2. * lame_mu_ * strain;
    const auto tmp = lame_lambda_ * dealii::trace(strain);
    for (int i = 0; i < DIM; ++i)
      stress[i][i] += tmp;
  }

private:
  double E_;
  double nu_;
  double lame_lambda_;
  double lame_mu_;
};

template <typename number>
class StiffnessOperator
    : public dealii::MatrixFreeOperators::Base<
          DIM, dealii::LinearAlgebra::distributed::Vector<number>> {

public:
  using FECellIntegrator = dealii::FEEvaluation<DIM, 1, 2, DIM, number>;
  using FECellDensityIntegrator = dealii::FEEvaluation<DIM, 0, 1, 1, number>;

  StiffnessOperator() : stress_(1., 0.3) {}

  void
  set_elastic_modulus(const dealii::LinearAlgebra::distributed::Vector<number>
                          &elastic_modulus) {
    const unsigned int n_cells = this->data->n_cell_batches();
    FECellIntegrator phi(*this->data);
    process_local_rho.resize(n_cells);
    const double Emin = 1.e-9;
    const double Emax = 1.;

    FECellDensityIntegrator rho(*this->data, 1);

    for (unsigned int cell = 0; cell < n_cells; ++cell) {
      rho.reinit(cell);
      rho.read_dof_values_plain(elastic_modulus);
      rho.evaluate(dealii::EvaluationFlags::values);
      process_local_rho[cell] = Emin + (Emax - Emin) * rho.get_value(0);
    }
  }
  void clear() override {
    dealii::MatrixFreeOperators::Base<
        DIM, dealii::LinearAlgebra::distributed::Vector<number>>::clear();
  }

  void
  initialize_dof_vector(dealii::LinearAlgebra::distributed::Vector<number> &vec,
                        const unsigned int dh_index) {
    this->data->initialize_dof_vector(vec, dh_index);
  }

  void local_vmult(FECellIntegrator &phi) const {
    const unsigned int cell = phi.get_current_cell_index();
    dealii::SymmetricTensor<2, DIM, dealii::VectorizedArray<number>> stress;
    phi.evaluate(dealii::EvaluationFlags::gradients);
    for (const auto q : phi.quadrature_point_indices()) {
      stress = phi.get_symmetric_gradient(q);
      stress_.convert_strain_to_stress<dealii::VectorizedArray<number>>(stress);
      phi.submit_symmetric_gradient(process_local_rho[cell] * stress, q);
    }
    phi.integrate(dealii::EvaluationFlags::gradients);
  }

  virtual void compute_diagonal() override {
    this->inverse_diagonal_entries.reset(
        new dealii::DiagonalMatrix<
            dealii::LinearAlgebra::distributed::Vector<number>>());
    dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
        this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal, 0);

    dealii::MatrixFreeTools::compute_diagonal(
        *this->data, inverse_diagonal, &StiffnessOperator<number>::local_vmult,
        this);

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (auto &diagonal_element : inverse_diagonal) {
      diagonal_element = (std::abs(diagonal_element) > 1.0e-10)
                             ? (1.0 / diagonal_element)
                             : 1.0;
    }
  }

  void
  local_apply(const dealii::MatrixFree<DIM, number> &data,
              dealii::LinearAlgebra::distributed::Vector<number> &dst,
              const dealii::LinearAlgebra::distributed::Vector<number> &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const {

    FECellIntegrator phi(data, 0);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
      phi.reinit(cell);
      phi.read_dof_values(src);
      local_vmult(phi);
      phi.distribute_local_to_global(dst);
    }
  }

  virtual void
  apply_add(dealii::LinearAlgebra::distributed::Vector<number> &dst,
            const dealii::LinearAlgebra::distributed::Vector<number> &src)
      const override {
    this->data->cell_loop(&StiffnessOperator<number>::local_apply, this, dst,
                          src);
  }

private:
  dealii::AlignedVector<dealii::VectorizedArray<number>> process_local_rho;
  Stress stress_;
};

class TopOpt : public ParOptProblem {

public:
  TopOpt(Teuchos::RCP<Teuchos::ParameterList> &params);
  void setup_system();
  void assemble_sensitivity_vector(LA::MPI::Vector &sensitivity);
  void assemble_rhs();
  void run();

  void solve_matrix_free(VectorType &solution,
                         const LA::MPI::Vector &elastic_modulus,
                         const VectorType &rhs);
  void write_statistics();
  void scale_design_to_satisfy_constraint(LA::MPI::Vector &design) const;

  void write_current_visualization();

  void output_results(const std::string &prefix, const VectorType &solution,
                      const std::map<std::string, Teuchos::RCP<LA::MPI::Vector>>
                          &density_vectors,
                      const unsigned int cycle);

  void checkpoint(const std::string &filename_prefix,
                  const LA::MPI::Vector &design,
                  const LA::MPI::Vector &material);

  void optimize_paropt(const unsigned int max_iters);

  void postprocess(const LA::MPI::Vector &gray_material);

  void optimize();

  ParOptQuasiDefMat *createQuasiDefMat();
  void getVarsAndBounds(ParOptVec *xvec, ParOptVec *lbvec, ParOptVec *ubvec);
  int evalObjCon(ParOptVec *xvec, ParOptScalar *fobj, ParOptScalar *cons);
  int evalObjConGradient(ParOptVec *xvec, ParOptVec *gvec, ParOptVec **Ac);

  void check_gradients();

private:
  LA::MPI::Vector design_lb;
  LA::MPI::Vector design_ub;

  double vol_frac_ = 0.5;
  double target_volume_;

  int optimization_iteration_ = 0;

  LA::MPI::Vector design_;
  LA::MPI::Vector material_;
  LA::MPI::Vector design_old_;

  MPI_Comm mpi_communicator_;
  dealii::ConditionalOStream pcout;
  dealii::TimerOutput timer_;

  Teuchos::RCP<Teuchos::ParameterList> params_;

  dealii::parallel::distributed::Triangulation<DIM> triangulation_;
  dealii::FESystem<DIM> fe_;
  dealii::DoFHandler<DIM> dof_handler_;

  dealii::FE_DGQ<DIM> density_fe_;
  dealii::DoFHandler<DIM> density_dof_handler_;

  VectorType solution_;
  VectorType lagrange_multiplier_vector_;
  VectorType system_rhs_;

  Teuchos::RCP<StiffnessOperator<double>> stiffness_operator_;

  LA::MPI::Vector sensitivity_;
  LA::MPI::Vector is_design_;

  dealii::AffineConstraints<double> constraints_;

  const double Emin = 1.e-9;
  const double Emax = 1.;

  std::string output_dir_;

  dealii::TableHandler statistics_;
  dealii::TableHandler statistics_postprocess_;

  double obj_scale_ = 1.;

  Stress stress_calculator_;
  Teuchos::RCP<NonlocalModelBase> nonlocal_model_;

  Teuchos::Array<double> boundary_displacements_;
  LA::MPI::Vector cell_at_boundary_;

  bool resume_ = false;

  bool set_objective_scale_ = true;

  bool write_results_ = true;

  std::chrono::steady_clock::time_point wall_clock_time_ =
      std::chrono::steady_clock::now();

  double total_time_ = 0.;

  double compliance_lb_ = 1.;

  std::vector<std::pair<double, std::string>> times_and_names_;

  dealii::MappingQ1<DIM> mapping_;
  dealii::AffineConstraints<double> empty_constraints_;

  dealii::MGConstrainedDoFs mg_constrained_dofs;
  using LevelMatrixType = StiffnessOperator<float>;
  dealii::MGLevelObject<LevelMatrixType> mg_matrices_;
  dealii::LinearAlgebra::distributed::Vector<double> elastic_modulus_;
  dealii::MGLevelObject<dealii::LinearAlgebra::distributed::Vector<float>>
      mg_densities_;
  dealii::MGTransferMatrixFree<DIM, float> mg_transfer_;

  dealii::DiagonalMatrix<VectorType> mass_matrix_;
};
