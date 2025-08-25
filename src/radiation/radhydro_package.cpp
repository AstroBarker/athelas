/**
 * @file radhydro_package.cpp
 * --------------
 *
 * @brief Radiation hydrodynamics package
 */
#include <limits>

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "geometry/grid.hpp"
#include "opacity/opac_variant.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_utilities.hpp"
#include "radiation/radhydro_package.hpp"
#include "solvers/root_finders.hpp"
#include "utils/abstractions.hpp"

namespace radiation {
using fluid::numerical_flux_gudonov_positivity;

RadHydroPackage::RadHydroPackage(const ProblemIn* /*pin*/, int n_stages,
                                 EOS* eos, Opacity* opac,
                                 ModalBasis* fluid_basis, ModalBasis* rad_basis,
                                 BoundaryConditions* bcs, double cfl, int nx,
                                 bool active)
    : active_(active), nx_(nx), cfl_(cfl), eos_(eos), opac_(opac),
      fluid_basis_(fluid_basis), rad_basis_(rad_basis), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", nx + 2 + 1, 5),
      u_f_l_("hydro::u_f_l_", nx + 2, 5), u_f_r_("hydro::u_f_r_", nx + 2, 5),
      flux_u_("hydro::flux_u_", n_stages + 1, nx + 2 + 1),
      scratch_k_("scratch_k_", nx + 2, fluid_basis_->get_order(), 5),
      scratch_km1_("scratch_km1_", nx + 2, fluid_basis_->get_order(), 5),
      scratch_sol_("scratch_k_", nx + 2, fluid_basis_->get_order(), 5) {
} // Need long term solution for flux_u_

KOKKOS_FUNCTION
void RadHydroPackage::update_explicit(const View3D<double> state,
                                      View3D<double> dU,
                                      const GridStructure& grid,
                                      const TimeStepInfo& dt_info) const {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto& order = fluid_basis_->get_order();
  const auto& ilo = grid.get_ilo();
  const auto& ihi = grid.get_ihi();

  const auto stage = dt_info.stage;

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(state, &grid, rad_basis_, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(state, &grid, fluid_basis_, bcs_, {0, 2});

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "RadHydro :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {ihi + 1, order, NUM_VARS_}),
      KOKKOS_LAMBDA(const int ix, const int k, const int q) {
        dU(ix, k, q) = 0.0;
      });

  // --- radiation Increment : Divergence ---
  radhydro_divergence(state, dU, grid, stage);

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "RadHydro :: Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int ix, const int k) {
      for (int q = 0; q < NUM_VARS_; ++q) {
        if (q < 3) {
          dU(ix, k, q) /= (fluid_basis_->get_mass_matrix(ix, k));
        } else {
          dU(ix, k, q) /= (rad_basis_->get_mass_matrix(ix, k));
        }
      }
      });
} // update_explicit

/**
 * @brief radiation hydrodynamic implicit term
 * Computes dU from source terms
 **/
KOKKOS_FUNCTION
void RadHydroPackage::update_implicit(const View3D<double> state,
                                      View3D<double> dU,
                                      const GridStructure& grid,
                                      const TimeStepInfo& dt_info) const {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto& order = fluid_basis_->get_order();
  const auto& ilo = grid.get_ilo();
  const auto& ihi = grid.get_ihi();

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "RadHydro :: Implicit :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {ihi + 1, order, NUM_VARS_}),
      KOKKOS_LAMBDA(const int ix, const int k, const int q) {
        dU(ix, k, q) = 0.0;
      });

  Kokkos::parallel_for(
      "RadHydro :: Implicit",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        const auto state_i = Kokkos::subview(state, i, Kokkos::ALL, Kokkos::ALL);
        const auto [du1, du2, du3, du4] = radhydro_source(state_i, grid, i, k);
        dU(i, k, 1) = du1;
        dU(i, k, 2) = du2;
        dU(i, k, 3) = du3;
        dU(i, k, 4) = du4;
      });
} // update_implicit

KOKKOS_FUNCTION
void RadHydroPackage::update_implicit_iterative(const View3D<double> state,
                                                View3D<double> dU,
                                                const GridStructure& grid,
                                                const TimeStepInfo& dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto& order = fluid_basis_->get_order();
  const auto& ilo = grid.get_ilo();
  const auto& ihi = grid.get_ihi();

  Kokkos::parallel_for(
      "RadHydro :: implicit iterative", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int iX) {
        auto state_i = Kokkos::subview(state, iX, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_ix =
            Kokkos::subview(scratch_sol_, iX, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_ix_k =
            Kokkos::subview(scratch_k_, iX, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_ix_km1 =
            Kokkos::subview(scratch_km1_, iX, Kokkos::ALL, Kokkos::ALL);
        auto R_ix = Kokkos::subview(dU, iX, Kokkos::ALL, Kokkos::ALL);

        // TODO(astrobarker): invert loops
        for (int k = 0; k < order; ++k) {
          // set radhydro vars
          for (int i = 0; i < NUM_VARS_; ++i) {
            scratch_sol_ix_k(k, i) = state_i(k, i);
            scratch_sol_ix_km1(k, i) = state_i(k, i);
            scratch_sol_ix(k, i) = state_i(k, i);
          }
        }

        root_finders::fixed_point_radhydro(
            R_ix, dt_info.dt_a, scratch_sol_ix_k, scratch_sol_ix_km1,
            scratch_sol_ix, grid, fluid_basis_, rad_basis_, eos_, opac_, iX);

        for (int k = 0; k < order; ++k) {
          for (int q = 1; q < NUM_VARS_; ++q) {
            state(iX, k, q) = scratch_sol_ix(k, q);
          }
        }
      });

} // update_implicit_iterative

// Compute the divergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
KOKKOS_FUNCTION
void RadHydroPackage::radhydro_divergence(const View3D<double> state,
                                          View3D<double> dU,
                                          const GridStructure& grid,
                                          const int stage) const {
  const auto& nNodes = grid.get_n_nodes();
  const auto& order = rad_basis_->get_order();
  const auto& ilo = grid.get_ilo();
  const auto& ihi = grid.get_ihi();

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states

  Kokkos::parallel_for(
      "RadHydro :: Interface States", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(const int i) {
        for (int q = 0; q < 3; ++q) {
          u_f_l_(i, q) = fluid_basis_->basis_eval(state, i - 1, q, nNodes + 1);
          u_f_r_(i, q) = fluid_basis_->basis_eval(state, i, q, 0);
        }
        for (int q = 3; q < NUM_VARS_; ++q) {
          u_f_l_(i, q) = rad_basis_->basis_eval(state, i - 1, q, nNodes + 1);
          u_f_r_(i, q) = rad_basis_->basis_eval(state, i, q, 0);
        }
      });

  // --- Calc numerical flux at all faces ---
  Kokkos::parallel_for(
      "RadHydro :: Numerical Fluxes", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(int iX) {
        auto lambda = nullptr;
        const double Pgas_L =
            pressure_from_conserved(eos_, u_f_l_(iX, 0), u_f_l_(iX, 1), u_f_l_(iX, 2), lambda);
        const double Cs_L = sound_speed_from_conserved(eos_, u_f_l_(iX, 0), u_f_l_(iX, 1),
                                                       u_f_l_(iX, 2), lambda);

        const double Pgas_R =
            pressure_from_conserved(eos_, u_f_r_(iX, 0), u_f_r_(iX, 1), u_f_r_(iX, 2), lambda);
        const double Cs_R = sound_speed_from_conserved(eos_, u_f_r_(iX, 0), u_f_r_(iX, 1),
                                                       u_f_r_(iX, 2), lambda);

        const double E_L = u_f_l_(iX, 3);
        const double F_L = u_f_l_(iX, 4);
        const double E_R = u_f_r_(iX, 3);
        const double F_R = u_f_r_(iX, 4);

        const double Prad_L = compute_closure(E_L, F_L);
        const double Prad_R = compute_closure(E_R, F_R);

        // --- Numerical Fluxes ---
        static constexpr double c2 = constants::c_cgs * constants::c_cgs;

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( u_f_l_(iX,  1 ), u_f_r_(iX,  1
        // ), P_L, P_R, lam_L, lam_R);
        const auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
            u_f_l_(iX, 0), u_f_r_(iX, 0), u_f_l_(iX, 1), u_f_r_(iX, 1), Pgas_L, Pgas_R, Cs_L, Cs_R);
        flux_u_(stage, iX) = flux_u;

        const double vstar = flux_u;
        // auto [flux_e, flux_f] =
        //    numerical_flux_hll_rad( E_L, E_R, F_L, F_R, P_L, P_R, vstar );
        const double eddington_factor = Prad_L / E_L;
        const double alpha =
            (constants::c_cgs - vstar) * std::sqrt(eddington_factor);
        const double flux_e = llf_flux(F_R, F_L, E_R, E_L, alpha);
        const double flux_f =
            llf_flux(c2 * Prad_R, c2 * Prad_L, F_R, F_L, alpha);

        const double advective_flux_e =
            (vstar >= 0) ? vstar * E_L : vstar * E_R;
        const double advective_flux_f =
            (vstar >= 0) ? vstar * F_L : vstar * F_R;

        dFlux_num_(iX, 0) = -flux_u_(stage, iX);
        dFlux_num_(iX, 1) = flux_p;
        dFlux_num_(iX, 2) = +flux_u_(stage, iX) * flux_p;

        dFlux_num_(iX, 3) = flux_e - advective_flux_e;
        dFlux_num_(iX, 4) = flux_f - advective_flux_f;
      });

  flux_u_(stage, ilo - 1) = flux_u_(stage, ilo);
  flux_u_(stage, ihi + 2) = flux_u_(stage, ihi + 1);

  // TODO(astrobarker): Is this pattern for the surface term okay?
  // --- Surface Term ---
    Kokkos::parallel_for(
        "RadHydro :: Surface Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ilo, 0, 0}, {ihi + 1, order, NUM_VARS_}),
        KOKKOS_CLASS_LAMBDA(const int iX, const int k, const int q) {
          const auto* basis = (q < 3) ? fluid_basis_ : rad_basis_;
          const auto& Poly_L = basis->get_phi(iX, 0, k);
          const auto& Poly_R = basis->get_phi(iX, nNodes + 1, k);
          const auto& X_L = grid.get_left_interface(iX);
          const auto& X_R = grid.get_left_interface(iX + 1);
          const auto& SqrtGm_L = grid.get_sqrt_gm(X_L);
          const auto& SqrtGm_R = grid.get_sqrt_gm(X_R);

          dU(iX, k, q) -= (+dFlux_num_(iX + 1, q) * Poly_R * SqrtGm_R -
                           dFlux_num_(iX + 0, q) * Poly_L * SqrtGm_L);
        });

  if (order > 1) {
    // --- Volume Term ---
    Kokkos::parallel_for(
        "RadHydro :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
        KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
          double local_sum1 = 0.0;
          double local_sum2 = 0.0;
          double local_sum3 = 0.0;
          double local_sum_e = 0.0;
          double local_sum_f = 0.0;
          const double vstar = flux_u_(stage, iX);
          for (int iN = 0; iN < nNodes; ++iN) {
            const double weight = grid.get_weights(iN);
            const double dphi_rad = rad_basis_->get_d_phi(iX, iN + 1, k);
            const double dphi_fluid = fluid_basis_->get_d_phi(iX, iN + 1, k);
            const double X = grid.node_coordinate(iX, iN);
            const double sqrt_gm = grid.get_sqrt_gm(X);

            auto lambda = nullptr;
            const double vel = fluid_basis_->basis_eval(state, iX, 1, iN + 1);
            const double P = pressure_from_conserved(
                eos_, fluid_basis_->basis_eval(state, iX, 0, iN + 1), vel,
                fluid_basis_->basis_eval(state, iX, 2, iN + 1), lambda);
            const double e_rad = rad_basis_->basis_eval(state, iX, 3, iN + 1);
            const double f_rad = rad_basis_->basis_eval(state, iX, 4, iN + 1);
            const double p_rad = compute_closure(e_rad, f_rad);
            const auto [flux1, flux2, flux3] = fluid::flux_fluid(vel, P);
            const auto [flux_e, flux_f] = flux_rad(e_rad, f_rad, p_rad, vstar);
            local_sum1 += weight * flux1 * dphi_fluid * sqrt_gm;
            local_sum2 += weight * flux2 * dphi_fluid * sqrt_gm;
            local_sum3 += weight * flux3 * dphi_fluid * sqrt_gm;
            local_sum_e += weight * flux_e * dphi_rad * sqrt_gm;
            local_sum_f += weight * flux_f * dphi_rad * sqrt_gm;
          }

          dU(iX, k, 0) += local_sum1;
          dU(iX, k, 1) += local_sum2;
          dU(iX, k, 2) += local_sum3;
          dU(iX, k, 3) += local_sum_e;
          dU(iX, k, 4) += local_sum_f;
        });
  }
} // radhydro_divergence

/**
 * @brief Compute source terms for radiation hydrodynamics system
 * @note Returns tuple<S_egas, S_vgas, S_erad, S_frad>
 *
 *   Note that here we take in a single 2D view state representing the radhydro
 *   state on a given cell. The indices are:
 *     0: fluid specific volume
 *     1: fluid velocity
 *     2: fluid total specific energy
 *     3: radiation energy density
 *     4: radiation flux F
 **/
auto RadHydroPackage::radhydro_source(const View2D<double> state,
                                      const GridStructure& grid, const int iX,
                                      const int k) const
    -> std::tuple<double, double, double, double> {
  return compute_increment_radhydro_source(state, k, grid, fluid_basis_,
                                           rad_basis_, eos_, opac_, iX);
}

// This is duplicate of above but used differently, in the root finder
// The code needs some refactoring in order to get rid of this version.
auto compute_increment_radhydro_source(const View2D<double> uCRH, const int k,
                                       const GridStructure& grid,
                                       const ModalBasis* fluid_basis,
                                       const ModalBasis* rad_basis,
                                       const EOS* eos, const Opacity* opac,
                                       const int iX)
    -> std::tuple<double, double, double, double> {
  constexpr static double c = constants::c_cgs;
  constexpr static double c2 = c * c;

  const int nNodes = grid.get_n_nodes();
  const double dx = grid.get_widths(iX);

  double local_sum_e_r = 0.0; // radiation energy source
  double local_sum_m_r = 0.0; // radiation momentum (flux) source
  double local_sum_e_g = 0.0; // gas energy source
  double local_sum_m_g = 0.0; // gas momentum (velocity) source
  for (int iN = 0; iN < nNodes; ++iN) {
    const double weight = grid.get_weights(iN);
    const double phi_rad = rad_basis->get_phi(iX, iN + 1, k);
    const double phi_fluid = fluid_basis->get_phi(iX, iN + 1, k);

    // Note: basis evaluations are awkward here.
    // must be sure to use the correct basis functions.
    const double tau = fluid_basis->basis_eval(uCRH, iX, 0, iN + 1);
    const double rho = 1.0 / tau;
    const double vel = fluid_basis->basis_eval(uCRH, iX, 1, iN + 1);
    const double em_t = fluid_basis->basis_eval(uCRH, iX, 2, iN + 1);

    auto lambda = nullptr;
    const double t_g = temperature_from_conserved(eos, tau, vel, em_t, lambda);

    // TODO(astrobarker): composition
    const double X = 1.0;
    const double Y = 1.0;
    const double Z = 1.0;

    const double kappa_r = rosseland_mean(opac, rho, t_g, X, Y, Z, lambda);
    const double kappa_p = planck_mean(opac, rho, t_g, X, Y, Z, lambda);

    const double E_r = rad_basis->basis_eval(uCRH, iX, 3, iN + 1);
    const double F_r = rad_basis->basis_eval(uCRH, iX, 4, iN + 1);
    const double P_r = compute_closure(E_r, F_r);

    // 4 force
    const auto [G0, G] =
        radiation_four_force(rho, vel, t_g, kappa_r, kappa_p, E_r, F_r, P_r);
//    std::println("G0 G {} {} {} {} {}", G0, G, E_r, em_t, t_g);

    const double source_e_r = -c * G0;
    const double source_m_r = -c2 * G;
    const double source_e_g = c * G0;
    const double source_m_g = G;

    local_sum_e_r += weight * phi_rad * source_e_r;
    local_sum_m_r += weight * phi_rad * source_m_r;
    local_sum_e_g += weight * phi_fluid * source_e_g;
    local_sum_m_g += weight * phi_fluid * source_m_g;
  }
  // \Delta x / M_kk
  const double dx_o_mkk_fluid = dx / fluid_basis->get_mass_matrix(iX, k);
  const double dx_o_mkk_rad = dx / rad_basis->get_mass_matrix(iX, k);

  return {local_sum_m_g * dx_o_mkk_fluid, local_sum_e_g * dx_o_mkk_fluid,
          local_sum_e_r * dx_o_mkk_rad, local_sum_m_r * dx_o_mkk_rad};
}

/**
 * @brief explicit radiation hydrodynamic timestep restriction
 **/
KOKKOS_FUNCTION
auto RadHydroPackage::min_timestep(const View3D<double> /*state*/,
                                   const GridStructure& grid,
                                   const TimeStepInfo& /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  const int& ilo = grid.get_ilo();
  const int& ihi = grid.get_ihi();

  double dt_out = 0.0;
  Kokkos::parallel_reduce(
      "RadHydro::min_timestep", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix, double& lmin) {
        const double dr = grid.get_widths(ix);
        static constexpr double eigval = constants::c_cgs;
        const double dt_old = dr / eigval;

        lmin = std::min(dt_old, lmin);
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  return dt_out;
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::name() const noexcept
    -> std::string_view {
  return "RadHydro";
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::is_active() const noexcept
    -> bool {
  return active_;
}

KOKKOS_FUNCTION
void RadHydroPackage::set_active(const bool active) { active_ = active; }

[[nodiscard]] KOKKOS_FUNCTION auto
RadHydroPackage::get_flux_u(const int stage, const int ix) const -> double {
  return flux_u_(stage, ix);
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::get_fluid_basis() const
    -> const ModalBasis* {
  return fluid_basis_;
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::get_rad_basis() const
    -> const ModalBasis* {
  return rad_basis_;
}

} // namespace radiation
