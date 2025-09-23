#pragma once

#include "Kokkos_Macros.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"
#include "utils/constants.hpp"

namespace ni {

using bc::BoundaryConditions;

// FullTrapping
// Exp / Leakage
// Schwartz (Schwarz)
// Schwartz (different form)
enum class NiHeatingModel {
  FullTrapping,
  ExpDeposition,
  Schwartz // TODO(astrobarker): rename
};

inline auto parse_model(const std::string &model) -> NiHeatingModel {
  if (model == "full_trapping") {
    return NiHeatingModel::FullTrapping;
  }
  if (model == "exp_deposition") {
    return NiHeatingModel::ExpDeposition;
  }
  if (model == "schwartz") {
    return NiHeatingModel::Schwartz;
  }
  throw std::invalid_argument("Unknown NiHeatingModel: " + model);
}

class NiHeatingPackage {
 public:
  NiHeatingPackage(const ProblemIn *pin, ModalBasis *basis, double cfl,
                   bool active = true);

  KOKKOS_FUNCTION
  void update_explicit(const State *const state, View3D<double> dU,
                       const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;

  KOKKOS_FUNCTION
  template <NiHeatingModel Model>
  void ni_update(const View3D<double> ucf, CompositionData *comps,
                 View3D<double> dU, const GridStructure &grid,
                 const TimeStepInfo &dt_info) const;

  // TODO(astrobarker): rewrite to have only two exp
  KOKKOS_FORCEINLINE_FUNCTION
  static auto eps_nickel1(const double time) {
    return E_NI_ * std::exp(-time * LAMBDA_NI_) +
           E_CO_ * (std::exp(-time / TAU_CO_) - std::exp(-time * LAMBDA_NI_));
  }

  // NOTE: E_NI_ etc are energy release per gram per second
  KOKKOS_FORCEINLINE_FUNCTION
  static auto eps_nickel2(const double x_ni, const double x_co) -> double {
    return E_NI_ * x_ni + E_CO_ * x_co;
  }

  KOKKOS_FUNCTION
  template <NiHeatingModel Model>
  [[nodiscard]]
  auto deposition_function(int ix, int node) const -> double;

  KOKKOS_FORCEINLINE_FUNCTION
  static auto dtau(const double rho, const double kappa_gamma, const double dz)
      -> double {
    return -rho * kappa_gamma * dz;
  }

  [[nodiscard]] KOKKOS_FUNCTION auto
  min_timestep(const State *const /*state*/, const GridStructure & /*grid*/,
               const TimeStepInfo & /*dt_info*/) const -> double;

  [[nodiscard]] KOKKOS_FUNCTION auto name() const noexcept -> std::string_view;

  [[nodiscard]] KOKKOS_FUNCTION auto is_active() const noexcept -> bool;

  void fill_derived(State *state, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

  KOKKOS_FUNCTION
  void set_active(bool active);

 private:
  bool active_;
  NiHeatingModel model_;

  View2D<double> deposition_function_;

  double gval_; // constant gravity

  ModalBasis *basis_;

  double cfl_;

  // constants
  static constexpr double TAU_NI_ =
      8.77591 * constants::seconds_to_days; // seconds TODO(astrobarker): impact
                                            // of using simpler values?
  static constexpr double LAMBDA_NI_ = 1.0 / TAU_NI_;
  static constexpr double TAU_CO_ =
      111.4 * constants::seconds_to_days; // seconds (113.6?)
  static constexpr double LAMBDA_CO_ = 1.0 / TAU_CO_;
  static constexpr double E_NI_ = 3.9e10; // erg
  static constexpr double E_CO_ = 6.78e9; // erg
};

KOKKOS_FORCEINLINE_FUNCTION
auto kappa_gamma(const double ye) -> double { return 0.06 * ye; }

} // namespace ni
