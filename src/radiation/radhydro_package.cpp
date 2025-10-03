#include <limits>

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "geometry/grid.hpp"
#include "opacity/opac_variant.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_utilities.hpp"
#include "radiation/radhydro_package.hpp"
#include "utils/abstractions.hpp"

namespace radiation {
using athelas::fluid::numerical_flux_gudonov_positivity;

RadHydroPackage::RadHydroPackage(const ProblemIn *pin, int n_stages, EOS *eos,
                                 Opacity *opac, ModalBasis *fluid_basis,
                                 ModalBasis *rad_basis, BoundaryConditions *bcs,
                                 double cfl, int nx, bool active)
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
void RadHydroPackage::update_explicit(const State *const state,
                                      View3D<double> dU,
                                      const GridStructure &grid,
                                      const TimeStepInfo &dt_info) const {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &order = fluid_basis_->get_order();
  static constexpr int ilo = 1;
  static const auto &ihi = grid.get_ihi();

  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(ucf, &grid, rad_basis_, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(ucf, &grid, fluid_basis_, bcs_, {0, 2});

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
        // Cache mass matrix values to avoid repeated lookups
        const double fluid_mm = fluid_basis_->get_mass_matrix(ix, k);
        const double rad_mm = rad_basis_->get_mass_matrix(ix, k);

        // Process fluid variables (q=0,1,2)
        for (int q = 0; q < 3; ++q) {
          dU(ix, k, q) /= fluid_mm;
        }

        // Process radiation variables (q=3,4)
        for (int q = 3; q < NUM_VARS_; ++q) {
          dU(ix, k, q) /= rad_mm;
        }
      });

  // --- Increment from Geometry ---
  if (grid.do_geometry()) {
    const auto uaf = state->u_af();
    radhydro_geometry(ucf, uaf, dU, grid);
  }
} // update_explicit

/**
 * @brief radiation hydrodynamic implicit term
 * Computes dU from source terms
 **/
KOKKOS_FUNCTION
void RadHydroPackage::update_implicit(const State *const state,
                                      View3D<double> dU,
                                      const GridStructure &grid,
                                      const TimeStepInfo &dt_info) const {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &order = fluid_basis_->get_order();
  static constexpr int ilo = 1;
  static const auto &ihi = grid.get_ihi();

  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

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
        const auto ucf_i = Kokkos::subview(ucf, i, Kokkos::ALL, Kokkos::ALL);
        const auto [du1, du2, du3, du4] =
            radhydro_source(state, ucf_i, grid, i, k);
        dU(i, k, 1) = du1;
        dU(i, k, 2) = du2;
        dU(i, k, 3) = du3;
        dU(i, k, 4) = du4;
      });
} // update_implicit

KOKKOS_FUNCTION
void RadHydroPackage::update_implicit_iterative(const State *const state,
                                                View3D<double> dU,
                                                const GridStructure &grid,
                                                const TimeStepInfo &dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &order = fluid_basis_->get_order();
  static constexpr int ilo = 1;
  static const auto &ihi = grid.get_ihi();

  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  Kokkos::parallel_for(
      "RadHydro :: implicit iterative", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        const auto ucf_i = Kokkos::subview(ucf, ix, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_ix =
            Kokkos::subview(scratch_sol_, ix, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_ix_k =
            Kokkos::subview(scratch_k_, ix, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_ix_km1 =
            Kokkos::subview(scratch_km1_, ix, Kokkos::ALL, Kokkos::ALL);
        const auto R_ix = Kokkos::subview(dU, ix, Kokkos::ALL, Kokkos::ALL);

        for (int k = 0; k < order; ++k) {
          // set radhydro vars
          for (int i = 0; i < NUM_VARS_; ++i) {
            scratch_sol_ix_k(k, i) = ucf_i(k, i);
            scratch_sol_ix_km1(k, i) = ucf_i(k, i);
            scratch_sol_ix(k, i) = ucf_i(k, i);
          }
        }

        fixed_point_radhydro(R_ix, dt_info.dt_a, scratch_sol_ix_k,
                             scratch_sol_ix_km1, scratch_sol_ix, state, grid,
                             fluid_basis_, rad_basis_, eos_, opac_, ix);

        for (int k = 0; k < order; ++k) {
          for (int q = 1; q < NUM_VARS_; ++q) {
            ucf(ix, k, q) = scratch_sol_ix(k, q);
          }
        }
      });

} // update_implicit_iterative

// Compute the divergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
KOKKOS_FUNCTION
void RadHydroPackage::radhydro_divergence(const State *const state,
                                          View3D<double> dU,
                                          const GridStructure &grid,
                                          const int stage) const {
  const auto u_stages = state->u_cf_stages();

  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  const auto uaf = state->u_af();

  const auto &nNodes = grid.get_n_nodes();
  const auto &order = rad_basis_->get_order();
  static constexpr int ilo = 1;
  static const auto &ihi = grid.get_ihi();

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states

  Kokkos::parallel_for(
      "RadHydro :: Interface States", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(const int i) {
        const int nnp1 = nNodes + 1;
        for (int q = 0; q < 3; ++q) {
          u_f_l_(i, q) = fluid_basis_->basis_eval(ucf, i - 1, q, nnp1);
          u_f_r_(i, q) = fluid_basis_->basis_eval(ucf, i, q, 0);
        }
        for (int q = 3; q < NUM_VARS_; ++q) {
          u_f_l_(i, q) = rad_basis_->basis_eval(ucf, i - 1, q, nnp1);
          u_f_r_(i, q) = rad_basis_->basis_eval(ucf, i, q, 0);
        }
      });

  // --- Calc numerical flux at all faces ---
  Kokkos::parallel_for(
      "RadHydro :: Numerical Fluxes", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        const double Pgas_L = uaf(ix - 1, nNodes + 1, 0);
        const double Cs_L = uaf(ix - 1, nNodes + 1, 2);

        const double Pgas_R = uaf(ix, 0, 0);
        const double Cs_R = uaf(ix, 0, 2);

        const double E_L = u_f_l_(ix, 3);
        const double F_L = u_f_l_(ix, 4);
        const double E_R = u_f_r_(ix, 3);
        const double F_R = u_f_r_(ix, 4);

        const double Prad_L = compute_closure(E_L, F_L);
        const double Prad_R = compute_closure(E_R, F_R);

        // --- Numerical Fluxes ---
        static constexpr double c2 = constants::c_cgs * constants::c_cgs;

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( u_f_l_(ix,  1 ),
        // u_f_r_(ix,  1
        // ), P_L, P_R, lam_L, lam_R);
        const auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
            u_f_l_(ix, 0), u_f_r_(ix, 0), u_f_l_(ix, 1), u_f_r_(ix, 1), Pgas_L,
            Pgas_R, Cs_L, Cs_R);
        flux_u_(stage, ix) = flux_u;

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

        dFlux_num_(ix, 0) = -flux_u;
        ;
        dFlux_num_(ix, 1) = flux_p;
        dFlux_num_(ix, 2) = +flux_u * flux_p;

        dFlux_num_(ix, 3) = flux_e - advective_flux_e;
        dFlux_num_(ix, 4) = flux_f - advective_flux_f;
      });

  flux_u_(stage, ilo - 1) = flux_u_(stage, ilo);
  flux_u_(stage, ihi + 2) = flux_u_(stage, ihi + 1);

  // TODO(astrobarker): Is this pattern for the surface term okay?
  // --- Surface Term ---
  Kokkos::parallel_for(
      "RadHydro :: Surface Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ilo, 0, 0},
                                             {ihi + 1, order, NUM_VARS_}),
      KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
        const auto *const basis = (q < 3) ? fluid_basis_ : rad_basis_;
        const auto &Poly_L = basis->get_phi(ix, 0, k);
        const auto &Poly_R = basis->get_phi(ix, nNodes + 1, k);
        const auto &X_L = grid.get_left_interface(ix);
        const auto &X_R = grid.get_left_interface(ix + 1);
        const auto &SqrtGm_L = grid.get_sqrt_gm(X_L);
        const auto &SqrtGm_R = grid.get_sqrt_gm(X_R);

        dU(ix, k, q) -= (+dFlux_num_(ix + 1, q) * Poly_R * SqrtGm_R -
                         dFlux_num_(ix + 0, q) * Poly_L * SqrtGm_L);
      });

  if (order > 1) [[likely]] {
    // --- Volume Term ---
    Kokkos::parallel_for(
        "RadHydro :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
        KOKKOS_CLASS_LAMBDA(const int ix, const int k) {
          double local_sum1 = 0.0;
          double local_sum2 = 0.0;
          double local_sum3 = 0.0;
          double local_sum_e = 0.0;
          double local_sum_f = 0.0;
          const double vstar = flux_u_(stage, ix);
          for (int iN = 0; iN < nNodes; ++iN) {
            const double weight = grid.get_weights(iN);
            const double dphi_rad = rad_basis_->get_d_phi(ix, iN + 1, k);
            const double dphi_fluid = fluid_basis_->get_d_phi(ix, iN + 1, k);
            const double X = grid.node_coordinate(ix, iN);
            const double sqrt_gm = grid.get_sqrt_gm(X);

            const double P = uaf(ix, iN + 1, 0);
            const double vel = fluid_basis_->basis_eval(ucf, ix, 1, iN + 1);
            const double e_rad = rad_basis_->basis_eval(ucf, ix, 3, iN + 1);
            const double f_rad = rad_basis_->basis_eval(ucf, ix, 4, iN + 1);
            const double p_rad = compute_closure(e_rad, f_rad);
            const auto [flux1, flux2, flux3] =
                athelas::fluid::flux_fluid(vel, P);
            const auto [flux_e, flux_f] = flux_rad(e_rad, f_rad, p_rad, vstar);
            local_sum1 += weight * flux1 * dphi_fluid * sqrt_gm;
            local_sum2 += weight * flux2 * dphi_fluid * sqrt_gm;
            local_sum3 += weight * flux3 * dphi_fluid * sqrt_gm;
            local_sum_e += weight * flux_e * dphi_rad * sqrt_gm;
            local_sum_f += weight * flux_f * dphi_rad * sqrt_gm;
          }

          dU(ix, k, 0) += local_sum1;
          dU(ix, k, 1) += local_sum2;
          dU(ix, k, 2) += local_sum3;
          dU(ix, k, 3) += local_sum_e;
          dU(ix, k, 4) += local_sum_f;
        });
  }
} // radhydro_divergence

/**
 * @brief Compute source terms for radiation hydrodynamics system
 * @note Returns tuple<S_egas, S_vgas, S_erad, S_frad>
 **/
auto RadHydroPackage::radhydro_source(const State *const state,
                                      const View2D<double> uCRH,
                                      const GridStructure &grid, const int ix,
                                      const int k) const
    -> std::tuple<double, double, double, double> {
  return compute_increment_radhydro_source(uCRH, k, state, grid, fluid_basis_,
                                           rad_basis_, eos_, opac_, ix);
}

// This is duplicate of above but used differently, in the root finder
// The code needs some refactoring in order to get rid of this version.
auto compute_increment_radhydro_source(
    const View2D<double> uCRH, const int k, const State *const state,
    const GridStructure &grid, const ModalBasis *fluid_basis,
    const ModalBasis *rad_basis, const EOS *eos, const Opacity *opac,
    const int ix) -> std::tuple<double, double, double, double> {
  constexpr static double c = constants::c_cgs;
  constexpr static double c2 = c * c;
  static const bool ionization_enabled = state->ionization_enabled();

  const int nNodes = grid.get_n_nodes();
  const double dx = grid.get_widths(ix);

  double local_sum_e_r = 0.0; // radiation energy source
  double local_sum_m_r = 0.0; // radiation momentum (flux) source
  double local_sum_e_g = 0.0; // gas energy source
  double local_sum_m_g = 0.0; // gas momentum (velocity) source
  for (int iN = 0; iN < nNodes; ++iN) {
    const double weight = grid.get_weights(iN);
    const double phi_rad = rad_basis->get_phi(ix, iN + 1, k);
    const double phi_fluid = fluid_basis->get_phi(ix, iN + 1, k);

    // Note: basis evaluations are awkward here.
    // must be sure to use the correct basis functions.
    const double tau = fluid_basis->basis_eval(uCRH, ix, 0, iN + 1);
    const double rho = 1.0 / tau;
    const double vel = fluid_basis->basis_eval(uCRH, ix, 1, iN + 1);
    const double em_t = fluid_basis->basis_eval(uCRH, ix, 2, iN + 1);

    double lambda[8];
    if (ionization_enabled) {
      paczynski_terms(state, ix, iN + 1, lambda);
    }
    const double t_g = temperature_from_conserved(eos, tau, vel, em_t, lambda);

    // TODO(astrobarker): composition
    const double X = 1.0;
    const double Y = 1.0;
    const double Z = 1.0;

    const double kappa_r = rosseland_mean(opac, rho, t_g, X, Y, Z, lambda);
    const double kappa_p = planck_mean(opac, rho, t_g, X, Y, Z, lambda);

    const double E_r = rad_basis->basis_eval(uCRH, ix, 3, iN + 1);
    const double F_r = rad_basis->basis_eval(uCRH, ix, 4, iN + 1);
    const double P_r = compute_closure(E_r, F_r);

    // 4 force
    const auto [G0, G] =
        radiation_four_force(rho, vel, t_g, kappa_r, kappa_p, E_r, F_r, P_r);

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
  const double dx_o_mkk_fluid = dx / fluid_basis->get_mass_matrix(ix, k);
  const double dx_o_mkk_rad = dx / rad_basis->get_mass_matrix(ix, k);

  return {local_sum_m_g * dx_o_mkk_fluid, local_sum_e_g * dx_o_mkk_fluid,
          local_sum_e_r * dx_o_mkk_rad, local_sum_m_r * dx_o_mkk_rad};
}

/**
 * @brief geometric source terms
 *
 * NOTE: identical to fluid_geometry. Should reduce overlap.
 * TODO(astrobarker): get rid of duplicate code with Hydro
 */
KOKKOS_FUNCTION
void RadHydroPackage::radhydro_geometry(const View3D<double> ucf,
                                        const View3D<double> uaf,
                                        View3D<double> dU,
                                        const GridStructure &grid) const {
  const int &nNodes = grid.get_n_nodes();
  const int &order = fluid_basis_->get_order();
  static constexpr int ilo = 1;
  static const int &ihi = grid.get_ihi();

  Kokkos::parallel_for(
      "Hydro :: Geometry Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int ix, const int k) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double P = uaf(ix, iN + 1, 0);

          const double X = grid.node_coordinate(ix, iN);

          local_sum += grid.get_weights(iN) * P *
                       fluid_basis_->get_phi(ix, iN + 1, k) * X;
        }

        dU(ix, k, 1) += (2.0 * local_sum * grid.get_widths(ix)) /
                        fluid_basis_->get_mass_matrix(ix, k);
      });
}

/**
 * @brief explicit radiation hydrodynamic timestep restriction
 **/
KOKKOS_FUNCTION
auto RadHydroPackage::min_timestep(const State *const /*ucf*/,
                                   const GridStructure &grid,
                                   const TimeStepInfo & /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  static constexpr int ilo = 1;
  static const int &ihi = grid.get_ihi();

  double dt_out = 0.0;
  Kokkos::parallel_reduce(
      "RadHydro::min_timestep", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix, double &lmin) {
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

/**
 * @brief fill RadHydro derived quantities for output
 *
 * TODO(astrobarker): extend
 */
void RadHydroPackage::fill_derived(State *state, const GridStructure &grid,
                                   const TimeStepInfo &dt_info) const {
  const int stage = dt_info.stage;

  auto u_s = state->u_cf_stages();

  auto uCF = Kokkos::subview(u_s, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  // hacky
  if (stage == -1) {
    uCF = state->u_cf();
  }
  auto uPF = state->u_pf();
  auto uAF = state->u_af();

  static constexpr int ilo = 0;
  static const int ihi = grid.get_ihi() + 2;
  const int nNodes = grid.get_n_nodes();
  static const bool ionization_enabled = state->ionization_enabled();

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(uCF, &grid, rad_basis_, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(uCF, &grid, fluid_basis_, bcs_, {0, 2});

  if (state->composition_enabled()) {
    fill_derived_comps(state, &grid, fluid_basis_);
  }

  if (ionization_enabled) {
    fill_derived_ionization(state, &grid, fluid_basis_);
  }

  Kokkos::parallel_for(
      "RadHydro::fill_derived", Kokkos::RangePolicy<>(ilo, ihi),
      KOKKOS_CLASS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes + 2; ++iN) {
          const double tau = fluid_basis_->basis_eval(uCF, ix, 0, iN);
          const double vel = fluid_basis_->basis_eval(uCF, ix, 1, iN);
          const double emt = fluid_basis_->basis_eval(uCF, ix, 2, iN);

          // const double e_rad = rad_basis_->basis_eval(uCF, ix, 3, iN + 1);
          // const double f_rad = rad_basis_->basis_eval(uCF, ix, 4, iN + 1);

          // const double flux_fact = flux_factor(e_rad, f_rad);

          const double rho = 1.0 / tau;
          const double momentum = rho * vel;
          const double sie = (emt - 0.5 * vel * vel);

          // This is probably not the cleanest logic, but setups with
          // ionization enabled and Paczynski disbled are an outlier.
          double lambda[8];
          if (ionization_enabled) {
            paczynski_terms(state, ix, iN, lambda);
          }
          const double pressure =
              pressure_from_conserved(eos_, tau, vel, emt, lambda);
          const double t_gas =
              temperature_from_conserved(eos_, tau, vel, emt, lambda);
          const double cs =
              sound_speed_from_conserved(eos_, tau, vel, emt, lambda);

          uPF(ix, iN, 0) = rho;
          uPF(ix, iN, 1) = momentum;
          uPF(ix, iN, 2) = sie;

          uAF(ix, iN, 0) = pressure;
          uAF(ix, iN, 1) = t_gas;
          uAF(ix, iN, 2) = cs;
        }
      });
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
    -> const ModalBasis * {
  return fluid_basis_;
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::get_rad_basis() const
    -> const ModalBasis * {
  return rad_basis_;
}

} // namespace radiation
