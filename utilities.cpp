#include "utilities.hpp"

#include <deal.II/grid/tria.h>

#include <deal.II/boost_adaptors/bounding_box.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/numerics/rtree.h>

#include <set>
#include <vector>

double setup_filter_matrix(const dealii::DoFHandler<DIM> &dof_handler,
                           LA::MPI::SparseMatrix &filter_matrix,
                           LA::MPI::Vector &is_design, const double radius,
                           const int filter_type, const Kernel &kernel,
                           MPI_Comm mpi_communicator) {

  std::vector<dealii::Point<DIM>> cell_centers_all_points(dof_handler.n_dofs());
  {
    LA::MPI::Vector cell_centers(dof_handler.locally_owned_dofs(),
                                 mpi_communicator);

    for (int d = 0; d < DIM; ++d) {
      for (const auto &cell : dof_handler.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
          const auto &c = cell->center();
          cell_centers[cell->dof_index(0)] = c[d];
        }
      }

      {
        dealii::Vector<double> cell_centers_all(cell_centers);
        for (unsigned int i = 0; i < cell_centers_all.size(); ++i) {
          cell_centers_all_points[i][d] = cell_centers_all[i];
        }
      }
    }
  }

  auto tree = dealii::pack_rtree_of_indices(cell_centers_all_points);

  std::vector<std::vector<unsigned int>> neighbor_dof_indices(
      dof_handler.n_locally_owned_dofs());

  std::vector<unsigned int> neighbors;

  unsigned int row = 0;

  dealii::Vector<float> is_design_all(is_design);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {

      if (filter_type == 1 && is_design[cell->dof_index(0)] < 0.5) {
        neighbor_dof_indices[row].push_back(cell->dof_index(0));
        ++row;
        continue;
      }

      neighbors.clear();

      auto lb = cell->center();
      auto ub = lb;
      for (int d = 0; d < DIM; ++d) {
        lb[d] -= radius;
        ub[d] += radius;
      }

      dealii::BoundingBox<DIM> box(std::make_pair(lb, ub));

      tree.query(boost::geometry::index::intersects(box),
                 std::back_inserter(neighbors));

      for (const auto n : neighbors) {
        if (cell_centers_all_points[n].distance(cell->center()) <= radius)
          if ((filter_type == 1 && is_design_all[n] > 0.5) || filter_type == 2)
            neighbor_dof_indices[row].push_back(n);
      }
      ++row;
    }
  }

  double avg_neighbors = 0;
  unsigned int max_neighbors = 0;
  for (const auto &a : neighbor_dof_indices) {
    avg_neighbors += a.size();
    max_neighbors =
        std::max(max_neighbors, static_cast<unsigned int>(a.size()));
  }

  MPI_Allreduce(MPI_IN_PLACE, &avg_neighbors, 1, MPI_DOUBLE, MPI_SUM,
                mpi_communicator);
  MPI_Allreduce(MPI_IN_PLACE, &max_neighbors, 1, MPI_UNSIGNED, MPI_MAX,
                mpi_communicator);

  std::set<unsigned int> set;

  dealii::IndexSet locally_relevant_density_dofs;

  for (const auto &n : neighbor_dof_indices) {
    for (const auto index : n)
      set.insert(index);
  }

  locally_relevant_density_dofs.set_size(dof_handler.n_dofs());
  locally_relevant_density_dofs.add_indices(set.begin(), set.end());

  {
    dealii::TrilinosWrappers::SparsityPattern dsp(
        dof_handler.locally_owned_dofs(), mpi_communicator, max_neighbors);
    row = 0;
    for (const auto cell : dof_handler.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        dsp.add_entries(cell->dof_index(0), neighbor_dof_indices[row].begin(),
                        neighbor_dof_indices[row].end(), false);
        ++row;
      }
    }
    dsp.compress();
    filter_matrix.reinit(dsp);
  }

  double global_max_weights_sum = 0.;
  {
    unsigned int row = 0;
    std::vector<double> weights;
    double max_weights_sum = 0.;
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        weights.clear();
        for (const auto n : neighbor_dof_indices[row]) {

          const double w =
              kernel(cell_centers_all_points[n] - cell->center(), radius);

          assert(w >= 0.);
          weights.push_back(w);
        }

        const double weights_sum =
            std::accumulate(weights.cbegin(), weights.cend(), 0.);

        max_weights_sum = std::max(max_weights_sum, weights_sum);
        ++row;
      }
    }
    MPI_Allreduce(&max_weights_sum, &global_max_weights_sum, 1, MPI_DOUBLE,
                  MPI_MAX, mpi_communicator);
  }

  {
    // fill filter matrix
    filter_matrix = 0.;
    unsigned int row = 0;
    std::vector<double> weights;

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        weights.clear();
        for (const auto n : neighbor_dof_indices[row]) {
          const double w =
              kernel(cell_centers_all_points[n] - cell->center(), radius);
          assert(w >= 0.);
          weights.push_back(w);
        }

        double denominator;
        if (filter_type == 2 && is_design[cell->dof_index(0)] > 0.5)
          denominator = global_max_weights_sum;
        else {
          denominator = std::accumulate(weights.cbegin(), weights.cend(), 0.);
        }
        for (unsigned int j = 0; j < weights.size(); ++j)
          filter_matrix.set(cell->dof_index(0), neighbor_dof_indices[row][j],
                            weights[j] / denominator);
        ++row;
      }
    }
    filter_matrix.compress(dealii::VectorOperation::insert);
  }

  return avg_neighbors / dof_handler.n_dofs();
}

void create_triangulation(
    dealii::parallel::fullydistributed::Triangulation<DIM> &triangulation,
    const Teuchos::RCP<Teuchos::ParameterList> &params,
    MPI_Comm mpi_communicator) {

  const std::string mesh_filename = params->get<std::string>("mesh file name");
  const int nrefine = params->get<int>("global mesh refinement steps", 0);

  // create and partition serial triangulation and create description
  dealii::TriangulationDescription::Description<DIM, DIM> description =
      dealii::TriangulationDescription::Utilities::
          create_description_from_triangulation_in_groups<DIM, DIM>(
              [mesh_filename, nrefine](auto &tria_base) {
                dealii::GridIn<DIM> grid_in;
                grid_in.attach_triangulation(tria_base);
                grid_in.read(mesh_filename, dealii::GridIn<DIM>::Format::msh);
                tria_base.refine_global(nrefine);
              },
              [](auto &tria_base, const auto comm, const auto group_size) {
                dealii::GridTools::partition_triangulation(
                    dealii::Utilities::MPI::n_mpi_processes(comm), tria_base);
              },
              mpi_communicator,
              dealii::Utilities::MPI::n_mpi_processes(mpi_communicator));

  // finally create triangulation
  triangulation.create_triangulation(description);
}

void setup_constraints(dealii::AffineConstraints<double> &constraints,
                       const dealii::DoFHandler<DIM> &dof_handler,
                       const Teuchos::Array<double> &bcs, const double scale) {

  for (int i = 0; i < bcs.size(); i += 3) {
    const int boundary_id = round(bcs[i]);
    const int mask = round(bcs[i + 1]);
    const double value = bcs[i + 2];
    if (DIM == 2)
      dealii::VectorTools::interpolate_boundary_values(
          dof_handler, boundary_id,
          dealii::Functions::ConstantFunction<DIM>(scale * value, DIM),
          constraints, std::vector<bool>{mask / 10 % 10, mask % 10});
    else
      dealii::VectorTools::interpolate_boundary_values(
          dof_handler, boundary_id,
          dealii::Functions::ConstantFunction<DIM>(scale * value, DIM),
          constraints,
          std::vector<bool>{mask / 100 % 10, mask / 10 % 10, mask % 10});
  }
}

void remove_grayness(LA::MPI::Vector &material,
                     const LA::MPI::Vector &is_design,
                     const double volume_fraction) {
  const double domain_volume = is_design.l1_norm();
  double l1 = 0., l2 = 1.;

  double threshold;

  LA::MPI::Vector gray_density(material);
  gray_density = material;

  unsigned int iter = 0;
  unsigned int max_iters = 10000;

  while (std::abs(l2 - l1) / (l1 + l2) > 1.e-6) {
    threshold = 0.5 * (l1 + l2);
    for (const auto i : material.locally_owned_elements())
      material[i] = gray_density[i] > threshold ? 1. : 0.;

    const double vf = (material * is_design) / domain_volume;
    if (vf > volume_fraction)
      l1 = threshold;
    else
      l2 = threshold;

    if (iter > max_iters) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                 "max iters reached in remove_grayness");
    }

    ++iter;
  }
}

void perturb_vector(LA::MPI::Vector &vec, const LA::MPI::Vector &min,
                    const LA::MPI::Vector &max, const double scale) {
  std::srand(std::time(nullptr));
  for (const auto i : vec.locally_owned_elements()) {
    const double perturbation =
        scale * 2. * (((double)std::rand() / RAND_MAX) - 0.5);
    double xnew = vec[i] + perturbation;
    xnew = std::max(xnew, min[i]);
    vec[i] = std::min(xnew, max[i]);
  }
  vec.compress(dealii::VectorOperation::insert);
}

double calculate_grayness(const LA::MPI::Vector &density) {
  LA::MPI::Vector tmp(density);
  for (const auto i : density.locally_owned_elements()) {
    tmp[i] = 4. * density[i] * (1. - density[i]);
  }
  return tmp.mean_value();
}

void calculate_grayness_gradient(LA::MPI::Vector &grad,
                                 const LA::MPI::Vector &density) {

  for (const auto i : density.locally_owned_elements()) {
    grad[i] = 4. * (1. - 2. * density[i]) / density.size();
  }
}

void read_material_density_from_file(const dealii::DoFHandler<DIM> &dof_handler,
                                     LA::MPI::Vector &design,
                                     LA::MPI::Vector &material,
                                     const std::string &filename,
                                     MPI_Comm mpi_communicator) {

  std::vector<dealii::Point<DIM>> cell_centers;
  std::vector<double> cell_material;
  std::vector<double> cell_design;

  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    std::ifstream file(filename);

    // if line begins with #, then skip it
    std::string line;
    std::getline(file, line);
    if (line[0] == '#')
      std::getline(file, line);

    if (DIM == 2) {
      double x, y, d, m;
      while (file >> x >> y >> d >> m) {
        cell_centers.emplace_back(x, y);
        cell_material.push_back(m);
        cell_design.push_back(d);
      }
    } else if (DIM == 3) {
      double x, y, z, d, m;
      while (file >> x >> y >> z >> d >> m) {
        cell_centers.emplace_back(x, y, z);
        cell_material.push_back(m);
        cell_design.push_back(d);
      }
    }
    file.close();
  }
  cell_centers =
      dealii::Utilities::MPI::broadcast(mpi_communicator, cell_centers);
  cell_material =
      dealii::Utilities::MPI::broadcast(mpi_communicator, cell_material);
  cell_design =
      dealii::Utilities::MPI::broadcast(mpi_communicator, cell_design);

  auto tree = dealii::pack_rtree_of_indices(cell_centers);

  std::vector<unsigned int> neighbors;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      neighbors.clear();
      tree.query(boost::geometry::index::nearest(cell->center(), 1),
                 std::back_inserter(neighbors));
      assert(neighbors.size() == 1);

      material[cell->dof_index(0)] = cell_material[neighbors[0]];
      design[cell->dof_index(0)] = cell_design[neighbors[0]];
    }
  }
  design.compress(dealii::VectorOperation::insert);
  material.compress(dealii::VectorOperation::insert);
}
