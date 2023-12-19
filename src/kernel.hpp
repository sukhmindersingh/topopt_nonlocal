#pragma once

#include "global.hpp"
#include <cmath>
#include <deal.II/base/tensor.h>
#include <iostream>

class Kernel {
public:
  Kernel(const double p) : p(p) {}

  double operator()(const dealii::Tensor<1, DIM> &x,
                    const double radius) const {
    // return 1. - std::pow(x.norm() / radius, p);
    return std::pow(1. - x.norm() / radius, p);
  }

private:
  double p;
};
