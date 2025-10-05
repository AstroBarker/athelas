#pragma once

#include "Kokkos_Macros.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "compdata.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"
#include "utils/constants.hpp"

namespace athelas::nickel {

using bc::BoundaryConditions;

// FullTrapping
// Swartz
// Jeffery 1999
enum class NiHeatingModel { FullTrapping, Swartz, Jeffery };

inline auto parse_model(const std::string &model) -> NiHeatingModel {
  if (model == "full_trapping") {
    return NiHeatingModel::FullTrapping;
  }
  if (model == "jeffery") {
    return NiHeatingModel::Jeffery;
  }
  if (model == "swartz") {
    return NiHeatingModel::Swartz;
  }
  THROW_ATHELAS_ERROR("Unknown nickel heating model!");
}

class NickelHeatingPackage {
 public:
  NickelHeatingPackage(const ProblemIn *pin, basis::ModalBasis *basis,
                       bool active = true);

  KOKKOS_FUNCTION
  void update_explicit(const State *const state, View3D<double> dU,
                       const GridStructure &grid, const TimeStepInfo &dt_info);

  KOKKOS_FUNCTION
  template <NiHeatingModel Model>
  void ni_update(const View3D<double> ucf, atom::CompositionData *comps,
                 View3D<double> dU, const GridStructure &grid,
                 const TimeStepInfo &dt_info) const;

  // NOTE: E_LAMBDA_NI etc are energy release per gram per second
  KOKKOS_FORCEINLINE_FUNCTION
  static auto eps_nickel_cobalt(const double x_ni, const double x_co)
      -> double {
    return eps_nickel(x_ni) + eps_cobalt(x_co);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static auto eps_nickel(const double x_ni) -> double {
    return E_LAMBDA_NI * x_ni;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static auto eps_cobalt(const double x_co) -> double {
    return E_LAMBDA_CO * x_co;
  }

  KOKKOS_FUNCTION
  template <NiHeatingModel Model>
  [[nodiscard]]
  auto deposition_function(const View3D<double> ucf,
                           const atom::CompositionData *comps,
                           const GridStructure &grid, int ix, int node) const
      -> double;

  KOKKOS_FORCEINLINE_FUNCTION
  static auto dtau(const double rho, const double kappa_gamma, const double dz)
      -> double {
    return -rho * kappa_gamma * dz;
  }

  [[nodiscard]] KOKKOS_FUNCTION auto
  min_timestep(const State * /*state*/, const GridStructure & /*grid*/,
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
  View3D<double> tau_gamma_; // [nx][node][angle]
  View2D<double> int_etau_domega_; // integration of e^-tau dOmega

  basis::ModalBasis *basis_;

  // constants
  static constexpr double TAU_NI_ =
      8.764372373400 * constants::seconds_to_days; // seconds
  static constexpr double LAMBDA_NI_ = 1.0 / TAU_NI_;
  static constexpr double TAU_CO_ =
      111.4 * constants::seconds_to_days; // seconds (113.6?)
  static constexpr double LAMBDA_CO_ = 1.0 / TAU_CO_;
  // These are eps_x * lambda_x and have units of erg/g/s
  static constexpr double E_LAMBDA_NI = 3.94e10; // erg / g / s
  static constexpr double E_LAMBDA_CO = 6.78e9; // erg / g / s

  // Jeffery 1999
  // The following are fractions of decay energy that go into gammas (F_NI_*)
  // and into positrons (F_PE_*).
  static constexpr double F_PE_NI_ = 0.004;
  static constexpr double F_GM_NI_ = 0.996;
  static constexpr double F_PE_CO_ = 0.032;
  static constexpr double F_GM_CO_ = 0.968;
};

KOKKOS_FORCEINLINE_FUNCTION
auto kappa_gamma(const double ye) -> double {
  static constexpr double KAPPA_COEF_ = 0.06; // Swartz gray opacity coef
  return KAPPA_COEF_ * ye;
}

} // namespace athelas::nickel
