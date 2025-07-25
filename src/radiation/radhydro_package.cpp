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
      dFlux_num_("hydro::dFlux_num_", 5, nx + 2 + 1),
      u_f_l_("hydro::u_f_l_", 5, nx + 2), u_f_r_("hydro::u_f_r_", 5, nx + 2),
      flux_u_("hydro::flux_u_", n_stages + 1, nx + 2 + 1),
      scratch_k_("scratch_k_", nx + 2, 5, fluid_basis_->get_order()),
      scratch_km1_("scratch_km1_", nx + 2, 5, fluid_basis_->get_order()),
      scratch_sol_("scratch_k_", nx + 2, 5, fluid_basis_->get_order()) {
} // Need long term solution for flux_u_

KOKKOS_FUNCTION
void RadHydroPackage::update_explicit(const View3D<double> state,
                                      View3D<double> dU,
                                      const GridStructure& grid,
                                      const TimeStepInfo& dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto& order = fluid_basis_->get_order();
  const auto& ilo   = grid.get_ilo();
  const auto& ihi   = grid.get_ihi();

  const auto stage = dt_info.stage;

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(state, &grid, rad_basis_, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(state, &grid, fluid_basis_, bcs_, {0, 2});

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "RadHydro :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_LAMBDA(const int q, const int ix, const int k) {
        dU(q, ix, k) = 0.0;
      });

  // --- radiation Increment : Divergence ---
  radhydro_divergence(state, dU, grid, stage);

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "RadHydro :: Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int ix, const int k) {
        for (int q = 0; q < 3; ++q) {
          dU(q, ix, k) /= (fluid_basis_->get_mass_matrix(ix, k));
        }
        for (int q = 3; q < NUM_VARS_; ++q) {
          dU(q, ix, k) /= (rad_basis_->get_mass_matrix(ix, k));
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
                                      const TimeStepInfo& dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto& order = fluid_basis_->get_order();
  const auto& ilo   = grid.get_ilo();
  const auto& ihi   = grid.get_ihi();

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "RadHydro :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_LAMBDA(const int q, const int ix, const int k) {
        dU(q, ix, k) = 0.0;
      });

  Kokkos::parallel_for(
      "RadHydro :: Implicit",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        auto state_i = Kokkos::subview(state, Kokkos::ALL, i, Kokkos::ALL);
        const auto [du1, du2, du3, du4] = radhydro_source(state_i, grid, i, k);
        dU(1, i, k)                     = du1;
        dU(2, i, k)                     = du2;
        dU(3, i, k)                     = du3;
        dU(4, i, k)                     = du4;
      });
} // update_implicit

KOKKOS_FUNCTION
void RadHydroPackage::update_implicit_iterative(const View3D<double> state,
                                                View3D<double> dU,
                                                const GridStructure& grid,
                                                const TimeStepInfo& dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto& order = fluid_basis_->get_order();
  const auto& ilo   = grid.get_ilo();
  const auto& ihi   = grid.get_ihi();

  Kokkos::parallel_for(
      "RadHydro :: implicit iterative", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int iX) {
        auto state_i = Kokkos::subview(state, Kokkos::ALL, iX, Kokkos::ALL);
        auto scratch_sol_ix =
            Kokkos::subview(scratch_sol_, iX, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_ix_k =
            Kokkos::subview(scratch_k_, iX, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_ix_km1 =
            Kokkos::subview(scratch_km1_, iX, Kokkos::ALL, Kokkos::ALL);
        auto R_ix = Kokkos::subview(dU, Kokkos::ALL, iX, Kokkos::ALL);

        // TODO(astrobarker): invert loops
        for (int k = 0; k < order; ++k) {
          // set radhydro vars
          for (int i = 0; i < NUM_VARS_; ++i) {
            scratch_sol_ix_k(i, k)   = state_i(i, k);
            scratch_sol_ix_km1(i, k) = state_i(i, k);
            scratch_sol_ix(i, k)     = state_i(i, k);
          }
        }

        root_finders::fixed_point_radhydro(
            R_ix, dt_info.dt_a, scratch_sol_ix_k, scratch_sol_ix_km1,
            scratch_sol_ix, grid, fluid_basis_, rad_basis_, eos_, opac_, iX);

        // TODO(astrobarker): invert loops
        for (int k = 0; k < order; ++k) {
          for (int q = 1; q < NUM_VARS_; ++q) {
            state(q, iX, k) = scratch_sol_ix(q, k);
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
                                          const int stage) {
  const auto& nNodes = grid.get_n_nodes();
  const auto& order  = rad_basis_->get_order();
  const auto& ilo    = grid.get_ilo();
  const auto& ihi    = grid.get_ihi();

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states

  Kokkos::parallel_for(
      "RadHydro :: Interface States", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(const int i) {
        for (int q = 0; q < 3; ++q) {
          u_f_l_(q, i) = fluid_basis_->basis_eval(state, i - 1, q, nNodes + 1);
          u_f_r_(q, i) = fluid_basis_->basis_eval(state, i, q, 0);
        }
        for (int q = 3; q < NUM_VARS_; ++q) {
          u_f_l_(q, i) = rad_basis_->basis_eval(state, i - 1, q, nNodes + 1);
          u_f_r_(q, i) = rad_basis_->basis_eval(state, i, q, 0);
        }
      });

  // --- Calc numerical flux at all faces ---
  Kokkos::parallel_for(
      "RadHydro :: Numerical Fluxes", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(int iX) {
        // TODO(astrobarker) rename uCF_L, uCF_R
        auto uCF_L = Kokkos::subview(u_f_l_, Kokkos::ALL, iX);
        auto uCF_R = Kokkos::subview(u_f_r_, Kokkos::ALL, iX);

        auto lambda = nullptr;
        const double Pgas_L =
            pressure_from_conserved(eos_, uCF_L(0), uCF_L(1), uCF_L(2), lambda);
        const double Cs_L = sound_speed_from_conserved(eos_, uCF_L(0), uCF_L(1),
                                                       uCF_L(2), lambda);

        const double Pgas_R =
            pressure_from_conserved(eos_, uCF_R(0), uCF_R(1), uCF_R(2), lambda);
        const double Cs_R = sound_speed_from_conserved(eos_, uCF_R(0), uCF_R(1),
                                                       uCF_R(2), lambda);

        const double E_L = uCF_L(3);
        const double F_L = uCF_L(4);
        const double E_R = uCF_R(3);
        const double F_R = uCF_R(4);

        const double Prad_L = compute_closure(E_L, F_L);
        const double Prad_R = compute_closure(E_R, F_R);

        // --- Numerical Fluxes ---
        static constexpr double c2 = constants::c_cgs * constants::c_cgs;

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( uCF_L( 1 ), uCF_R( 1
        // ), P_L, P_R, lam_L, lam_R);
        auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
            uCF_L(0), uCF_R(0), uCF_L(1), uCF_R(1), Pgas_L, Pgas_R, Cs_L, Cs_R);
        flux_u_(stage, iX) = flux_u;

        const double vstar = flux_u;
        // auto [flux_e, flux_f] =
        //    numerical_flux_hll_rad( E_L, E_R, F_L, F_R, P_L, P_R, vstar );
        const double eddington_factor = Prad_L / E_L;
        const double alpha =
            (constants::c_cgs - vstar) * std::sqrt(eddington_factor);
        auto flux_e = llf_flux(F_R, F_L, E_R, E_L, alpha);
        auto flux_f = llf_flux(c2 * Prad_R, c2 * Prad_L, F_R, F_L, alpha);
        double advective_flux_e = 0.0;
        double advective_flux_f = 0.0;

        advective_flux_e = (vstar >= 0) ? vstar * E_L : vstar * E_R;
        advective_flux_f = (vstar >= 0) ? vstar * F_L : vstar * F_R;

        dFlux_num_(0, iX) = -flux_u_(stage, iX);
        dFlux_num_(1, iX) = flux_p;
        dFlux_num_(2, iX) = +flux_u_(stage, iX) * flux_p;

        dFlux_num_(3, iX) = flux_e - advective_flux_e;
        dFlux_num_(4, iX) = flux_f - advective_flux_f;
      });

  flux_u_(stage, ilo - 1) = flux_u_(stage, ilo);
  flux_u_(stage, ihi + 2) = flux_u_(stage, ihi + 1);

  // TODO(astrobarker): Is this pattern for the surface term okay?
  // --- Surface Term ---
  for (int q = 0; q < NUM_VARS_; ++q) {
    const auto* basis = (q < 3) ? fluid_basis_ : rad_basis_;
    Kokkos::parallel_for(
        "RadHydro :: Surface Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
        KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
          const auto& Poly_L   = basis->get_phi(iX, 0, k);
          const auto& Poly_R   = basis->get_phi(iX, nNodes + 1, k);
          const auto& X_L      = grid.get_left_interface(iX);
          const auto& X_R      = grid.get_left_interface(iX + 1);
          const auto& SqrtGm_L = grid.get_sqrt_gm(X_L);
          const auto& SqrtGm_R = grid.get_sqrt_gm(X_R);

          dU(q, iX, k) -= (+dFlux_num_(q, iX + 1) * Poly_R * SqrtGm_R -
                           dFlux_num_(q, iX + 0) * Poly_L * SqrtGm_L);
        });
  }

  if (order > 1) {
    // --- Volume Term ---
    Kokkos::parallel_for(
        "RadHydro :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
        KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
          double local_sum1  = 0.0;
          double local_sum2  = 0.0;
          double local_sum3  = 0.0;
          double local_sum_e = 0.0;
          double local_sum_f = 0.0;
          for (int iN = 0; iN < nNodes; ++iN) {
            auto lambda      = nullptr;
            const double vel = fluid_basis_->basis_eval(state, iX, 1, iN + 1);
            const double P   = pressure_from_conserved(
                eos_, fluid_basis_->basis_eval(state, iX, 0, iN + 1), vel,
                fluid_basis_->basis_eval(state, iX, 2, iN + 1), lambda);
            const double e_rad = rad_basis_->basis_eval(state, iX, 3, iN + 1);
            const double f_rad = rad_basis_->basis_eval(state, iX, 4, iN + 1);
            const double p_rad = compute_closure(e_rad, f_rad);
            const auto [flux1, flux2, flux3] = fluid::flux_fluid(vel, P);
            const auto [flux_e, flux_f] =
                flux_rad(e_rad, f_rad, p_rad, flux_u_(stage, iX));
            const double X = grid.node_coordinate(iX, iN);
            local_sum1 += grid.get_weights(iN) * flux1 *
                          fluid_basis_->get_d_phi(iX, iN + 1, k) *
                          grid.get_sqrt_gm(X);
            local_sum2 += grid.get_weights(iN) * flux2 *
                          fluid_basis_->get_d_phi(iX, iN + 1, k) *
                          grid.get_sqrt_gm(X);
            local_sum3 += grid.get_weights(iN) * flux3 *
                          fluid_basis_->get_d_phi(iX, iN + 1, k) *
                          grid.get_sqrt_gm(X);
            local_sum_e += grid.get_weights(iN) * flux_e *
                           rad_basis_->get_d_phi(iX, iN + 1, k) *
                           grid.get_sqrt_gm(X);
            local_sum_f += grid.get_weights(iN) * flux_f *
                           rad_basis_->get_d_phi(iX, iN + 1, k) *
                           grid.get_sqrt_gm(X);
          }

          dU(0, iX, k) += local_sum1;
          dU(1, iX, k) += local_sum2;
          dU(2, iX, k) += local_sum3;
          dU(3, iX, k) += local_sum_e;
          dU(4, iX, k) += local_sum_f;
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
  constexpr static double c  = constants::c_cgs;
  constexpr static double c2 = c * c;

  const int nNodes = grid.get_n_nodes();

  double local_sum_e_r = 0.0; // radiation energy source
  double local_sum_m_r = 0.0; // radiation momentum (flux) source
  double local_sum_e_g = 0.0; // gas energy source
  double local_sum_m_g = 0.0; // gas momentum (velocity) source
  for (int iN = 0; iN < nNodes; ++iN) {
    // Note: basis evaluations are awkward here.
    // must be sure to use the correct basis functions.
    const double tau  = fluid_basis->basis_eval(uCRH, iX, 0, iN + 1);
    const double rho  = 1.0 / tau;
    const double vel  = fluid_basis->basis_eval(uCRH, iX, 1, iN + 1);
    const double em_t = fluid_basis->basis_eval(uCRH, iX, 2, iN + 1);

    auto lambda      = nullptr;
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

    const double source_e_r = -c * G0;
    const double source_m_r = -c2 * G;
    const double source_e_g = c * G0;
    const double source_m_g = G;

    local_sum_e_r +=
        grid.get_weights(iN) * rad_basis->get_phi(iX, iN + 1, k) * source_e_r;
    local_sum_m_r +=
        grid.get_weights(iN) * rad_basis->get_phi(iX, iN + 1, k) * source_m_r;
    local_sum_e_g +=
        grid.get_weights(iN) * fluid_basis->get_phi(iX, iN + 1, k) * source_e_g;
    local_sum_m_g +=
        grid.get_weights(iN) * fluid_basis->get_phi(iX, iN + 1, k) * source_m_g;
  }
  // \Delta x / M_kk
  const double dx_o_mkk_fluid =
      grid.get_widths(iX) / fluid_basis->get_mass_matrix(iX, k);
  const double dx_o_mkk_rad =
      grid.get_widths(iX) / rad_basis->get_mass_matrix(iX, k);

  return {local_sum_m_g * dx_o_mkk_fluid, local_sum_e_g * dx_o_mkk_fluid,
          local_sum_e_r * dx_o_mkk_rad, local_sum_m_r * dx_o_mkk_rad};
}

/**
 * @brief explicit radiation hydrodynamic timestep restriction
 **/
KOKKOS_FUNCTION
auto RadHydroPackage::min_timestep(const View3D<double> state,
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
        const double dr                = grid.get_widths(ix);
        static constexpr double eigval = constants::c_cgs;
        const double dt_old            = dr / eigval;

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
