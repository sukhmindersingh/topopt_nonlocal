#pragma once

#include <cmath>
#include <deal.II/base/tensor.h>

#include "global.hpp"

class Kernel {
public:
  Kernel(const double p = 1.) : p(p) {}

  double operator()(const dealii::Tensor<1, DIM> &x,
                    const double radius) const {
    return std::pow(1. - x.norm() / radius, p);
  }

private:
  double p;
};
