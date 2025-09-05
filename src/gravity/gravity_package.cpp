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
#include "pgen/problem_in.hpp"
#include "utils/abstractions.hpp"

namespace gravity {

GravityPackage::GravityPackage(const ProblemIn* /*pin*/, GravityModel model,
                               const double gval, ModalBasis* basis,
                               const double cfl, const bool active)
    : active_(active), model_(model), gval_(gval), basis_(basis), cfl_(cfl) {}

KOKKOS_FUNCTION
void GravityPackage::update_explicit(const View3D<double> state,
                                     View3D<double> dU,
                                     const GridStructure& grid,
                                     const TimeStepInfo& /*dt_info*/) const {
  if (model_ == GravityModel::Spherical) {
    gravity_update<GravityModel::Spherical>(state, dU, grid);
  } else [[unlikely]] {
    gravity_update<GravityModel::Constant>(state, dU, grid);
  }
}

KOKKOS_FUNCTION
template <GravityModel Model>
void GravityPackage::gravity_update(const View3D<double> state,
                                    View3D<double> dU,
                                    const GridStructure& grid) const {
  const int& nNodes = grid.get_n_nodes();
  const int& order = basis_->get_order();
  static constexpr int ilo = 1;
  static const int& ihi = grid.get_ihi();

  // This can probably be simplified.
  Kokkos::parallel_for(
      "Gravity :: Update",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int ix, const int k) {
        double local_sum_v = 0.0;
        double local_sum_e = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(ix, iN);
          const double sqrt_gm = grid.get_sqrt_gm(X);
          const double weight = grid.get_weights(iN);
          if constexpr (Model == GravityModel::Spherical) {
            local_sum_v += weight * basis_->get_phi(ix, iN + 1, k) *
                           grid.enclosed_mass(ix, iN) * sqrt_gm /
                           ((X * X) * basis_->basis_eval(state, ix, 0, iN + 1));
            local_sum_e +=
                local_sum_v * basis_->basis_eval(state, ix, 1, iN + 1);
          } else {
            local_sum_v += sqrt_gm * weight * basis_->get_phi(ix, iN + 1, k) *
                           gval_ / basis_->basis_eval(state, ix, 0, iN + 1);
            local_sum_e +=
                local_sum_v * basis_->basis_eval(state, ix, 1, iN + 1);
          }
        }

        dU(ix, k, 1) -=
            (constants::G_GRAV * local_sum_v * grid.get_widths(ix)) /
            basis_->get_mass_matrix(ix, k);
        dU(ix, k, 2) -=
            (constants::G_GRAV * local_sum_e * grid.get_widths(ix)) /
            basis_->get_mass_matrix(ix, k);
      });
}

/**
 * @brief Gravitational timestep restriction
 * @note This just returns max dt
 **/
KOKKOS_FUNCTION
auto GravityPackage::min_timestep(const View3D<double> /*state*/,
                                  const GridStructure& /*grid*/,
                                  const TimeStepInfo& /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = std::numeric_limits<double>::max() / 100.0;
  static constexpr double dt_out = MAX_DT;
  return dt_out;
}

void GravityPackage::fill_derived(State* /*state*/,
                                  const GridStructure& /*grid*/) const {}

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

} // namespace gravity
