#pragma once

#include "global.hpp"
#include "kernel.hpp"

#include <deal.II/grid/tria.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/boost_adaptors/bounding_box.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/rtree.h>
#include <deal.II/numerics/vector_tools.h>

#include <set>
#include <vector>


void create_triangulation(
    dealii::parallel::fullydistributed::Triangulation<DIM> &triangulation,
    const Teuchos::RCP<Teuchos::ParameterList> &params,
    MPI_Comm mpi_communicator);


double setup_filter_matrix(const dealii::DoFHandler<DIM> &dof_handler,
                           LA::MPI::SparseMatrix &filter_matrix,
                           LA::MPI::Vector &is_filtered, const double radius,
                           const int filter_type, const Kernel &kernel,
                           MPI_Comm mpi_communicator);

void setup_constraints(dealii::AffineConstraints<double> &constraints,
                       const dealii::DoFHandler<DIM> &dof_handler,
                       const Teuchos::Array<double> &bcs,
                       const double scale = 1.);

void remove_grayness(LA::MPI::Vector &material,
                     const LA::MPI::Vector &is_design,
                     const double volume_fraction);

void perturb_vector(LA::MPI::Vector &vec, const LA::MPI::Vector &min,
                    const LA::MPI::Vector &max, const double scale);

double calculate_grayness(const LA::MPI::Vector &density);

void calculate_grayness_gradient(LA::MPI::Vector &grad,
                                 const LA::MPI::Vector &density);

void read_material_density_from_file(const dealii::DoFHandler<DIM> &dof_handler,
                                     LA::MPI::Vector &design,
                                     LA::MPI::Vector &material,
                                     const std::string &filename,
                                     MPI_Comm mpi_communicator);

class FilterOperator {

public:
  virtual void vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Not implemented!");
  }
  virtual void Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) {

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Not implemented!");
  }
  virtual void Tvmult_add(LA::MPI::Vector &dst, const LA::MPI::Vector &src) {

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Not implemented!");
  }
};

class FilterMatrix : public FilterOperator {

public:
  LA::MPI::SparseMatrix &get_filter_matrix() { return filter_matrix; }

  void vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) {
    filter_matrix.vmult(dst, src);
  }
  void Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) {
    filter_matrix.Tvmult(dst, src);
  }
  void Tvmult_add(LA::MPI::Vector &dst, const LA::MPI::Vector &src) {
    filter_matrix.Tvmult_add(dst, src);
  }

private:
  LA::MPI::SparseMatrix filter_matrix;
};

class FilterMatrixFree : public FilterOperator {

public:
  FilterMatrixFree(const dealii::DoFHandler<DIM> &dh,
                   const LA::MPI::Vector &is_design, const double filter_r,
                   const int filter_type, const Kernel &kernel,
                   MPI_Comm mpi_communicator)
      : dof_handler(dh), is_design_(is_design), kernel_(kernel),
        pcout(std::cout,
              dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
        timer_(mpi_communicator, pcout, dealii::TimerOutput::summary,
               dealii::TimerOutput::wall_times) {

    radius_ = filter_r;
    filter_type_ = filter_type;
    std::vector<LA::MPI::Vector> cell_centers;

    for (int d = 0; d < DIM; ++d)
      cell_centers.emplace_back(dof_handler.locally_owned_dofs(),
                                mpi_communicator);

    cell_centers_all_points.resize(dof_handler.n_dofs());

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        const auto &c = cell->center();
        for (int d = 0; d < DIM; ++d)
          cell_centers[d][cell->dof_index(0)] = c[d];
      }
    }

    {
      std::vector<dealii::Vector<double>> cell_centers_all;

      for (int d = 0; d < DIM; ++d) {
        cell_centers_all.emplace_back(cell_centers[d]);
      }

      for (unsigned int i = 0; i < cell_centers_all[0].size(); ++i) {
        for (int d = 0; d < DIM; ++d) {
          cell_centers_all_points[i][d] = cell_centers_all[d][i];
        }
      }
    }

    denominator.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);

    neighbors_.resize(dof_handler.n_locally_owned_dofs());
    const auto tree = dealii::pack_rtree_of_indices(cell_centers_all_points);
    dealii::Vector<double> is_design_all(is_design_);

    denominator = 0.;

    double sum_num_neighbors = 0.;

    std::vector<typename dealii::DoFHandler<DIM>::active_cell_iterator>
        cell_iterators;
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        cell_iterators.push_back(cell);
      }
    }

#pragma omp parallel for
    for (auto row = 0; row < cell_iterators.size(); ++row) {
      const auto &cell = cell_iterators[row];
      const auto center = cell->center();
      const dealii::types::global_dof_index dof_index = cell->dof_index(0);

      if (filter_type_ == 1 && is_design_[cell->dof_index(0)] < 0.5) {
        neighbors_[row].push_back(dof_index);
      } else {
        std::vector<unsigned int> neighbors_tmp;
        auto lb = center;
        auto ub = center;
        for (int d = 0; d < DIM; ++d) {
          lb[d] -= radius_;
          ub[d] += radius_;
        }
        dealii::BoundingBox<DIM> box(std::make_pair(lb, ub));
        tree.query(boost::geometry::index::intersects(box),
                   std::back_inserter(neighbors_tmp));
        for (const auto n : neighbors_tmp) {
          if (cell_centers_all_points[n].distance(center) <= radius_) {
            if ((filter_type_ == 1 && is_design_all[n] > 0.5) ||
                filter_type_ == 2) {
              neighbors_[row].push_back(n);
            }
          }
        }
      }
    }
  }

  void set_parameters(const double radius, const int filter_type) {
    radius_ = radius;
    filter_type_ = filter_type;
  }

  void vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) {
    vmult(dst, src, false);
  }

  void vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src,
             const bool transpose = false) {

    dealii::TimerOutput::Scope t(timer_, "filtering");

    dst = 0.;

    dealii::Vector<double> is_design_all(is_design_);

    LA::MPI::Vector rhs(src);
    rhs = src;

    if (transpose) {
      for (const auto i : src.locally_owned_elements())
        rhs[i] /= denominator[i];
    }

    dealii::Vector<double> rhs_full(rhs);

    denominator = 0.;

    double sum_num_neighbors = 0.;

    std::vector<typename dealii::DoFHandler<DIM>::active_cell_iterator>
        cell_iterators;
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        cell_iterators.push_back(cell);
      }
    }

    // #pragma omp parallel for
    // for (unsigned int cell_index = 0; cell_index < cell_iterators.size();
    //      ++cell_index) {
    // const auto &cell = cell_iterators[cell_index];

    // for (const auto &cell : dof_handler.active_cell_iterators()) {
    //   if (cell->is_locally_owned()) {

    using Iterator = decltype(dof_handler.begin_active());

    struct ScratchData {};
    struct CopyData {};

    VectorType ds;
    ds.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorType dr;
    dr.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
    dr = 0.;
    ds = 0.;

#pragma omp parallel for
    for (unsigned int row = 0; row < cell_iterators.size(); ++row) {
      const auto &cell = cell_iterators[row];
      const auto center = cell->center();
      const dealii::types::global_dof_index dof_index = cell->dof_index(0);
      for (const auto n : neighbors_[row]) {
        // const double w = radius_ -
        // cell_centers_all_points[n].distance(center);
        const double w = kernel_(cell_centers_all_points[n] - center, radius_);
        ds[dof_index] += w * rhs_full[n];
        dr[dof_index] += w;
      }
    }

    std::copy(dr.begin(), dr.end(), denominator.begin());
    std::copy(ds.begin(), ds.end(), dst.begin());

    MPI_Allreduce(MPI_IN_PLACE, &sum_num_neighbors, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    sum_num_neighbors /= dof_handler.n_dofs();
    if (first)
      pcout << "Filter radius: " << radius_
            << " Average number of neighbors: " << sum_num_neighbors
            << std::endl;

    // dst.compress(dealii::VectorOperation::add);
    // denominator.compress(dealii::VectorOperation::add);
    const auto global_max_weights_sum = denominator.linfty_norm();

    if (filter_type_ == 2)
      for (const auto i : denominator.locally_owned_elements()) {
        denominator[i] =
            (is_design_[i] > 0.5) ? global_max_weights_sum : denominator[i];
      }

    if (!transpose) {
      for (const auto i : dst.locally_owned_elements())
        dst[i] /= denominator[i];
    }
    first = false;
  }

  void Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) {
    assert(first == false);
    vmult(dst, src, true);
  }

  void Tvmult_add(LA::MPI::Vector &dst, const LA::MPI::Vector &src) {
    assert(first == false);
    LA::MPI::Vector tmp(src);
    vmult(tmp, src, true);
    dst += tmp;
  }

private:
  const dealii::DoFHandler<DIM> &dof_handler;
  int filter_type_;
  double radius_;
  const LA::MPI::Vector &is_design_;
  std::vector<dealii::Point<DIM>> cell_centers_all_points;
  LA::MPI::Vector denominator;
  bool first = true;
  dealii::ConditionalOStream pcout;
  dealii::TimerOutput timer_;
  std::vector<std::vector<unsigned int>> neighbors_;
  Kernel kernel_;
};
