/**
 * @file gravity_package.cpp
 * --------------
 *
 * @brief Gravitational source package
 **/
#include <limits>

#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "gravity/gravity_package.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "utils/abstractions.hpp"

namespace athelas::gravity {

using basis::ModalBasis;

GravityPackage::GravityPackage(const ProblemIn * /*pin*/, GravityModel model,
                               const double gval, ModalBasis *basis,
                               const double cfl, const bool active)
    : active_(active), model_(model), gval_(gval), basis_(basis), cfl_(cfl) {}

void GravityPackage::update_explicit(const State *const state,
                                     View3D<double> dU,
                                     const GridStructure &grid,
                                     const TimeStepInfo &dt_info) const {
  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  if (model_ == GravityModel::Spherical) {
    gravity_update<GravityModel::Spherical>(ucf, dU, grid);
  } else [[unlikely]] {
    gravity_update<GravityModel::Constant>(ucf, dU, grid);
  }
}

template <GravityModel Model>
void GravityPackage::gravity_update(const View3D<double> state,
                                    View3D<double> dU,
                                    const GridStructure &grid) const {
  const int nNodes = grid.get_n_nodes();
  const int &order = basis_->get_order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);

  // This can probably be simplified.
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Update", DevExecSpace(), ib.s, ib.e,
      kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        double local_sum_v = 0.0;
        double local_sum_e = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double X = grid.node_coordinate(i, q);
          const double sqrt_gm = grid.get_sqrt_gm(X);
          const double weight = grid.get_weights(q);
          if constexpr (Model == GravityModel::Spherical) {
            local_sum_v += weight * basis_->get_phi(i, q + 1, k) *
                           grid.enclosed_mass(i, q) * sqrt_gm /
                           ((X * X) * basis_->basis_eval(state, i, 0, q + 1));
            local_sum_e += local_sum_v * basis_->basis_eval(state, i, 1, q + 1);
          } else {
            local_sum_v += sqrt_gm * weight * basis_->get_phi(i, q + 1, k) *
                           gval_ / basis_->basis_eval(state, i, 0, q + 1);
            local_sum_e += local_sum_v * basis_->basis_eval(state, i, 1, q + 1);
          }
        }

        dU(i, k, 1) -= (constants::G_GRAV * local_sum_v * grid.get_widths(i)) /
                       basis_->get_mass_matrix(i, k);
        dU(i, k, 2) -= (constants::G_GRAV * local_sum_e * grid.get_widths(i)) /
                       basis_->get_mass_matrix(i, k);
      });
}

/**
 * @brief Gravitational timestep restriction
 * @note This just returns max dt
 **/
KOKKOS_FUNCTION
auto GravityPackage::min_timestep(const State *const /*state*/,
                                  const GridStructure & /*grid*/,
                                  const TimeStepInfo & /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = std::numeric_limits<double>::max() / 100.0;
  static constexpr double dt_out = MAX_DT;
  return dt_out;
}

void GravityPackage::fill_derived(State * /*state*/,
                                  const GridStructure & /*grid*/,
                                  const TimeStepInfo & /*dt_info*/) const {}

[[nodiscard]] KOKKOS_FUNCTION auto GravityPackage::name() const noexcept
    -> std::string_view {
  return "Gravity";
}

[[nodiscard]] KOKKOS_FUNCTION auto GravityPackage::is_active() const noexcept
    -> bool {
  return active_;
}

KOKKOS_FUNCTION
void GravityPackage::set_active(const bool active) { active_ = active; }

} // namespace athelas::gravity
