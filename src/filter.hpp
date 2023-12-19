#pragma once

#include <Teuchos_TestForException.hpp>
#include <chrono>
#include <iostream>

#include "Teuchos_TestForException.hpp"
#include "global.hpp"
#include "kernel.hpp"
#include "utilities.hpp"
#include <Teuchos_ParameterList.hpp>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/dofs/dof_handler.h>

enum class PenalizationType { PowerLaw, Ramp };

inline double scaling(const double x, const double c) {
  return c + x * (1. - c);
}

inline double deriv_scaling(const double x, const double c) { return 1. - c; }

inline double heaviside(const double x, const double beta, const double eta) {
  return (std::tanh(beta * eta) + std::tanh(beta * (x - eta))) /
         (std::tanh(beta * eta) + std::tanh(beta * (1 - eta)));
}

inline double deriv_heaviside(const double x, const double beta,
                              const double eta) {
  return beta * (1. - std::pow(std::tanh(beta * (x - eta)), 2)) /
         (std::tanh(beta * eta) + std::tanh(beta * (1 - eta)));
}

inline double penalization(const double x, const double p,
                           const PenalizationType type) {
  switch (type) {
  case PenalizationType::PowerLaw:
    return std::pow(x, p);
  case PenalizationType::Ramp:
    return x / (1. + p * (1. - x));
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                               "Invalid penalization type");
  }
}

inline double deriv_penalization(const double x, const double p,
                                 const PenalizationType type) {
  switch (type) {
  case PenalizationType::PowerLaw:
    return p * std::pow(x, p - 1.);
  case PenalizationType::Ramp:
    return (1. + p) / std::pow(1. + p * (1. - x), 2);
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                               "Invalid penalization type");
  }
}

class NonlocalModelBase {

protected:
  Teuchos::RCP<LA::MPI::Vector> design_;
  Teuchos::RCP<LA::MPI::Vector> material_;
  Teuchos::RCP<LA::MPI::Vector> elastic_modulus_;
  Teuchos::RCP<LA::MPI::Vector> nonlocal_elastic_modulus_;
  Teuchos::RCP<LA::MPI::Vector> filtered_density_1_;
  Teuchos::RCP<LA::MPI::Vector> filtered_density_2_;
  Teuchos::RCP<LA::MPI::Vector> skeleton_;

  Teuchos::RCP<FilterOperator> material_filter_matrix_;
  Teuchos::RCP<FilterOperator> filter_matrix_;

  LA::MPI::Vector is_design_;
  LA::MPI::Vector cell_at_boundary_;
  LA::MPI::Vector cell_at_symmetry_boundary_;

  dealii::IndexSet locally_owned_elements_;
  double penalization_constant_;
  double beta_ = 0.;

  double zeta_;

  double volume_fraction_;

  Teuchos::RCP<Teuchos::ParameterList> params_;

  dealii::ConditionalOStream &pcout_;

  bool needs_continuation_ = false;

  PenalizationType penalization_type_;

  double cell_measure_;

public:
  NonlocalModelBase(Teuchos::RCP<Teuchos::ParameterList> &params,
                    const dealii::DoFHandler<DIM> &dof_handler,
                    const LA::MPI::Vector &is_design,
                    dealii::ConditionalOStream &pcout)
      : params_(params), is_design_(is_design), pcout_(pcout) {

    volume_fraction_ = params->get<double>("volume fraction");

    is_design_ = is_design;

    if (params_->get<std::string>("penalization type") == "power_law")
      penalization_type_ = PenalizationType::PowerLaw;
    else if (params_->get<std::string>("penalization type") == "ramp")
      penalization_type_ = PenalizationType::Ramp;

    zeta_ = params_->get<double>("zeta");

    double filter_r = 1.e-6;
    double material_filter_r = 1.e-6;

    double cell_edge_length;
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        cell_edge_length = cell->diameter() / std::sqrt(DIM);
        cell_measure_ = cell->measure();
        break;
      }
    }

    if (params_->get<std::string>("standard filter type") ==
        "discretization_based") {
      filter_r =
          params_->get<double>("standard filter radius") * cell_edge_length;
    } else if (params_->get<std::string>("standard filter type") ==
               "radius_based") {
      filter_r = params_->get<double>("standard filter radius");
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                                 "Invalid filter type");
    }

    material_filter_r = params_->get<double>("material filter radius");

    pcout_ << "cell edge length: " << cell_edge_length << std::endl;
    pcout_ << "standard filter radius: " << filter_r << std::endl;
    pcout_ << "material filter radius: " << material_filter_r << std::endl;

    const auto material_filter_behavior_type =
        params_->get<int>("material filter behavior type", 2);
    const auto standard_filter_behavior_type =
        params_->get<int>("standard filter behavior type", 1);

    Kernel material_filter_kernel(1.);

    if (!params_->get<bool>("use filter matrix free", true)) {
      filter_matrix_ = Teuchos::rcp(new FilterMatrix);
      auto avg_neighbors = setup_filter_matrix(
          dof_handler,
          Teuchos::rcp_dynamic_cast<FilterMatrix>(filter_matrix_)
              ->get_filter_matrix(),
          is_design_, filter_r, standard_filter_behavior_type, Kernel(1.),
          MPI_COMM_WORLD);
      pcout_ << "Avg. neighbors standard filter: " << avg_neighbors
             << std::endl;

      if ((std::abs(filter_r - material_filter_r) < 1.e-6) &&
          (material_filter_behavior_type == standard_filter_behavior_type))
        material_filter_matrix_ = filter_matrix_;
      else {
        material_filter_matrix_ = Teuchos::rcp(new FilterMatrix);
        avg_neighbors = setup_filter_matrix(
            dof_handler,
            Teuchos::rcp_dynamic_cast<FilterMatrix>(material_filter_matrix_)
                ->get_filter_matrix(),
            is_design_, material_filter_r, material_filter_behavior_type,
            material_filter_kernel, MPI_COMM_WORLD);
      }
      pcout_ << "Avg. neighbors material filter: " << avg_neighbors
             << std::endl;
    } else {
      filter_matrix_ = Teuchos::RCP(new FilterMatrixFree(
          dof_handler, is_design_, filter_r, standard_filter_behavior_type,
          Kernel(1.), MPI_COMM_WORLD));
      material_filter_matrix_ = Teuchos::RCP(
          new FilterMatrixFree(dof_handler, is_design_, material_filter_r,
                               material_filter_behavior_type,
                               material_filter_kernel, MPI_COMM_WORLD));
    }

    locally_owned_elements_ = dof_handler.locally_owned_dofs();

    penalization_constant_ = params_->get<double>("penalization constant");

    material_ = Teuchos::rcp(new LA::MPI::Vector(is_design_));
    design_ = Teuchos::rcp(new LA::MPI::Vector(is_design_));
    elastic_modulus_ = Teuchos::rcp(new LA::MPI::Vector(is_design_));
    nonlocal_elastic_modulus_ = Teuchos::rcp(new LA::MPI::Vector(is_design_));
    filtered_density_2_ = Teuchos::rcp(new LA::MPI::Vector(is_design_));
    filtered_density_1_ = Teuchos::rcp(new LA::MPI::Vector(is_design_));
    skeleton_ = Teuchos::rcp(new LA::MPI::Vector(is_design_));

    const auto betas =
        params_->get<Teuchos::Array<double>>("beta", Teuchos::Array<double>());
    beta_ = betas.begin() != betas.end() ? betas[0] : 0.;

    needs_continuation_ = false;
  }

  void set_peanlization_constant(const double penalization_constant) {
    penalization_constant_ = penalization_constant;
  }

  void increment_penalization_constant(const double increment) {
    penalization_constant_ += increment;
    pcout_ << "penalization constant incremented to " << penalization_constant_
           << std::endl;
  }

  bool needs_continuation() const { return needs_continuation_; }

  virtual void set_continuation_parameter(const double beta) { beta_ = beta; }

  virtual double get_continuation_parameter() const { return beta_; }

  virtual void set_design(const LA::MPI::Vector &design) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }

  virtual Teuchos::RCP<LA::MPI::Vector> get_elastic_modulus() {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }

  virtual Teuchos::RCP<LA::MPI::Vector> get_material() { return material_; }

  virtual Teuchos::RCP<LA::MPI::Vector> get_design() { return design_; }
  virtual Teuchos::RCP<LA::MPI::Vector> get_skeleton() { return skeleton_; }
  virtual Teuchos::RCP<LA::MPI::Vector> get_filtered_density_1() {
    return filtered_density_1_;
  }
  virtual Teuchos::RCP<LA::MPI::Vector> get_filtered_density_2() {
    return filtered_density_2_;
  }

  virtual void filter_sensitivity(LA::MPI::Vector &sensitivity) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }

  virtual void get_constraint_gradient(LA::MPI::Vector &grad) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }

  virtual double get_constraint_value() const {
    return *material_ * is_design_ - volume_fraction_ * is_design_.l1_norm();
  }

  virtual double get_volume_fraction() const {
    return (*material_ * is_design_) / is_design_.l1_norm();
  }

  virtual void get_volume_fraction_gradient(LA::MPI::Vector &grad) {
    get_constraint_gradient(grad);
    grad /= is_design_.l1_norm();
  }

  virtual double get_total_volume() const {

    return material_->l1_norm() * cell_measure_;
  }

  virtual double get_boundary_constraint_value(
      const LA::MPI::Vector &is_boundary_constrained) {
    return *filtered_density_2_ * is_boundary_constrained - 0.1;
  }

  virtual void get_boundary_constraint_gradient(
      LA::MPI::Vector &grad, const LA::MPI::Vector &is_boundary_constrained) {
    material_filter_matrix_->Tvmult(grad, is_boundary_constrained);
  }

  virtual void
  apply_raw_material_filter(LA::MPI::Vector &elastic_modulus,
                            const LA::MPI::Vector &material) const {
    material_filter_matrix_->vmult(elastic_modulus, material);
    for (const auto i : locally_owned_elements_)
      elastic_modulus[i] = material[i] * scaling(elastic_modulus[i], zeta_);
  }
};

class StandardSIMP : public NonlocalModelBase {

public:
  StandardSIMP(Teuchos::RCP<Teuchos::ParameterList> &params,
               const dealii::DoFHandler<DIM> &dof_handler,
               const LA::MPI::Vector &is_design,
               dealii::ConditionalOStream &pcout)
      : NonlocalModelBase(params, dof_handler, is_design, pcout) {}

  void set_design(const LA::MPI::Vector &design) override {
    *design_ = design;
    filter_matrix_->vmult(*material_, *design_);
  }

  Teuchos::RCP<LA::MPI::Vector> get_elastic_modulus() override {
    const double p = penalization_constant_;
    for (const auto i : locally_owned_elements_)
      (*elastic_modulus_)[i] =
          penalization((*material_)[i], p, penalization_type_);
    elastic_modulus_->compress(dealii::VectorOperation::insert);
    return elastic_modulus_;
  }

  void filter_sensitivity(LA::MPI::Vector &sensitivity) override {
    LA::MPI::Vector grad(sensitivity);
    const double p = penalization_constant_;
    LA::MPI::Vector tmp(sensitivity);
    for (const auto i : locally_owned_elements_)
      tmp[i] = deriv_penalization((*material_)[i], p, penalization_type_) *
               sensitivity[i];
    filter_matrix_->Tvmult(grad, tmp);
    sensitivity = grad;
  }

  void get_constraint_gradient(LA::MPI::Vector &grad) override {
    filter_matrix_->Tvmult(grad, is_design_);
  }
};

class SIMPHeaviside : public NonlocalModelBase {

private:
  double eta_;

public:
  SIMPHeaviside(Teuchos::RCP<Teuchos::ParameterList> &params,
                const dealii::DoFHandler<DIM> &dof_handler,
                const LA::MPI::Vector &is_design,
                dealii::ConditionalOStream &pcout)
      : NonlocalModelBase(params, dof_handler, is_design, pcout) {
    needs_continuation_ = true;
    eta_ = params_->get<double>("eta");
  }

  void set_design(const LA::MPI::Vector &design) override {
    const double beta = beta_;
    const double eta = eta_;
    *design_ = design;
    filter_matrix_->vmult(*filtered_density_1_, *design_);
    for (const auto i : locally_owned_elements_)
      (*material_)[i] = heaviside((*filtered_density_1_)[i], beta, eta);
  }

  Teuchos::RCP<LA::MPI::Vector> get_elastic_modulus() override {
    const double p = penalization_constant_;
    for (const auto i : locally_owned_elements_)
      (*elastic_modulus_)[i] =
          penalization((*material_)[i], p, penalization_type_);
    elastic_modulus_->compress(dealii::VectorOperation::insert);
    return elastic_modulus_;
  }

  void filter_sensitivity(LA::MPI::Vector &sensitivity) override {
    LA::MPI::Vector grad(sensitivity);
    const double p = penalization_constant_;
    LA::MPI::Vector tmp(sensitivity);
    for (const auto i : locally_owned_elements_)
      tmp[i] = deriv_penalization((*material_)[i], p, penalization_type_) *
               deriv_heaviside((*filtered_density_1_)[i], beta_, eta_) *
               sensitivity[i];
    filter_matrix_->Tvmult(grad, tmp);
    sensitivity = grad;
  }

  void get_constraint_gradient(LA::MPI::Vector &grad) override {
    LA::MPI::Vector tmp(grad);
    for (const auto i : locally_owned_elements_)
      tmp[i] = deriv_heaviside((*filtered_density_1_)[i], beta_, eta_) *
               is_design_[i];
    filter_matrix_->Tvmult(grad, tmp);
  }
};

class NonlocalModel : public NonlocalModelBase {

public:
  NonlocalModel(Teuchos::RCP<Teuchos::ParameterList> &params,
                const dealii::DoFHandler<DIM> &dof_handler,
                const LA::MPI::Vector &is_design,
                dealii::ConditionalOStream &pcout)
      : NonlocalModelBase(params, dof_handler, is_design, pcout) {}

  void set_design(const LA::MPI::Vector &design) override {
    *design_ = design;
    filter_matrix_->vmult(*material_, *design_);
    material_filter_matrix_->vmult(*filtered_density_2_, *design_);
  }

  Teuchos::RCP<LA::MPI::Vector> get_elastic_modulus() override {
    const double p = penalization_constant_;
    for (const auto i : locally_owned_elements_) {
      (*elastic_modulus_)[i] =
          penalization((*material_)[i], p, penalization_type_) *
          scaling((*filtered_density_2_)[i], zeta_);
    }
    elastic_modulus_->compress(dealii::VectorOperation::insert);
    return elastic_modulus_;
  }

  void filter_sensitivity(LA::MPI::Vector &sensitivity) override {

    LA::MPI::Vector grad(sensitivity);
    const double p = penalization_constant_;

    LA::MPI::Vector tmp(sensitivity);
    LA::MPI::Vector tmp2(sensitivity);

    for (const auto i : locally_owned_elements_) {
      tmp[i] = deriv_penalization((*material_)[i], p, penalization_type_) *
               scaling((*filtered_density_2_)[i], zeta_) * sensitivity[i];
    }

    filter_matrix_->Tvmult(grad, tmp);

    for (const auto i : locally_owned_elements_) {
      tmp[i] = penalization((*material_)[i], p, penalization_type_) *
               deriv_scaling((*filtered_density_2_)[i], zeta_) * sensitivity[i];
    }

    material_filter_matrix_->Tvmult_add(grad, tmp);

    sensitivity = grad;
  }

  void get_constraint_gradient(LA::MPI::Vector &grad) override {
    filter_matrix_->Tvmult(grad, is_design_);
  }
};

class NonlocalModelHeaviside : public NonlocalModelBase {

private:
  double eta_;

public:
  NonlocalModelHeaviside(Teuchos::RCP<Teuchos::ParameterList> &params,
                         const dealii::DoFHandler<DIM> &dof_handler,
                         const LA::MPI::Vector &is_design,
                         dealii::ConditionalOStream &pcout)
      : NonlocalModelBase(params, dof_handler, is_design, pcout) {
    eta_ = params_->get<double>("eta");
    needs_continuation_ = true;
  }

  void set_design(const LA::MPI::Vector &design) override {

    const double beta = beta_;
    const double eta = eta_;

    *design_ = design;
    filter_matrix_->vmult(*filtered_density_1_, *design_);

    for (const auto i : locally_owned_elements_)
      (*material_)[i] = heaviside((*filtered_density_1_)[i], beta, eta);

    material_filter_matrix_->vmult(*filtered_density_2_, *material_);
  }

  Teuchos::RCP<LA::MPI::Vector> get_elastic_modulus() override {
    const double p = penalization_constant_;
    const double beta = beta_;

    for (const auto i : locally_owned_elements_) {
      (*elastic_modulus_)[i] =
          penalization((*material_)[i], p, penalization_type_) *
          scaling((*filtered_density_2_)[i], zeta_);
    }

    elastic_modulus_->compress(dealii::VectorOperation::insert);
    return elastic_modulus_;
  }

  void filter_sensitivity(LA::MPI::Vector &sensitivity) override {
    LA::MPI::Vector grad(sensitivity);
    const double p1 = penalization_constant_;
    const double beta = beta_;

    LA::MPI::Vector tmp1(sensitivity);
    LA::MPI::Vector tmp2(sensitivity);

    for (const auto i : locally_owned_elements_) {
      tmp1[i] = deriv_penalization((*material_)[i], p1, penalization_type_) *
                deriv_heaviside((*filtered_density_1_)[i], beta, eta_) *
                scaling((*filtered_density_2_)[i], zeta_) * sensitivity[i];
    }

    filter_matrix_->Tvmult(grad, tmp1);

    for (const auto i : locally_owned_elements_) {
      tmp1[i] = penalization((*material_)[i], p1, penalization_type_) *
                deriv_scaling((*filtered_density_2_)[i], zeta_) *
                sensitivity[i];
    }

    material_filter_matrix_->Tvmult(tmp2, tmp1);
    for (const auto i : locally_owned_elements_) {
      tmp1[i] =
          deriv_heaviside((*filtered_density_1_)[i], beta, eta_) * tmp2[i];
    }
    filter_matrix_->Tvmult_add(grad, tmp1);
    sensitivity = grad;
  }

  void get_constraint_gradient(LA::MPI::Vector &grad) override {

    const double beta = beta_;

    LA::MPI::Vector tmp(grad);

    for (const auto i : locally_owned_elements_)
      tmp[i] = deriv_heaviside((*filtered_density_1_)[i], beta, eta_) *
               is_design_[i];

    filter_matrix_->Tvmult(grad, tmp);
  }
};
