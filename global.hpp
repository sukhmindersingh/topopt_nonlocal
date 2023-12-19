#pragma once

#if defined(PLANAR)
const int DIM = 2;
#else
const int DIM = 3;
#endif

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
// #include <deal.II/base/conditional_ostream.h>

namespace LA {
using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA

using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

// dealii::ConditionalOStream
// pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
