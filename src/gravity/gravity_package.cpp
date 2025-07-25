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
  const int& order  = basis_->get_order();
  const int& ilo    = grid.get_ilo();
  const int& ihi    = grid.get_ihi();

  // This can probably be simplified.
  Kokkos::parallel_for(
      "Gravity :: Spherical update",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
        double local_sum_v = 0.0;
        double local_sum_e = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(iX, iN);
          if constexpr (Model == GravityModel::Spherical) {
            local_sum_v += constants::G_GRAV * grid.get_weights(iN) *
                           basis_->get_phi(iX, iN + 1, k) *
                           grid.enclosed_mass(iX, iN) * grid.get_sqrt_gm(X) /
                           ((X * X) * basis_->basis_eval(state, iX, 0, iN + 1));
            local_sum_e +=
                local_sum_v * basis_->basis_eval(state, iX, 1, iN + 1);
          } else {
            local_sum_v +=
                grid.get_weights(iN) * basis_->get_phi(iX, iN + 1, k) * gval_ /
                basis_->basis_eval(state, iX, 0, iN + 1) * grid.get_sqrt_gm(X);
            local_sum_e +=
                local_sum_v * basis_->basis_eval(state, iX, 1, iN + 1);
          }
        }

        dU(1, iX, k) -= (local_sum_v * grid.get_widths(iX)) /
                        basis_->get_mass_matrix(iX, k);
        dU(2, iX, k) -= (local_sum_e * grid.get_widths(iX)) /
                        basis_->get_mass_matrix(iX, k);
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
