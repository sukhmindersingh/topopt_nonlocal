#pragma once

#if defined(PLANAR)
const int DIM = 2;
#else
const int DIM = 3;
#endif

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace LA {
using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA

using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
