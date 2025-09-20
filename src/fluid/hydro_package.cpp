/**
 * @file hydro_package.cpp
 * --------------
 *
 * @brief Pure hydrodynamics package
 */
#include <limits>

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "fluid/hydro_package.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "utils/abstractions.hpp"

namespace fluid {

HydroPackage::HydroPackage(const ProblemIn * /*pin*/, int n_stages, EOS *eos,
                           ModalBasis *basis, BoundaryConditions *bcs,
                           double cfl, int nx, bool active)
    : active_(active), nx_(nx), cfl_(cfl), eos_(eos), basis_(basis), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", nx + 2 + 1, 3),
      u_f_l_("hydro::u_f_l_", nx + 2, 3), u_f_r_("hydro::u_f_r_", nx + 2, 3),
      flux_u_("hydro::flux_u_", n_stages + 1, nx + 2 + 1) {
} // Need long term solution for flux_u_

KOKKOS_FUNCTION
void HydroPackage::update_explicit(const State *const state, View3D<double> dU,
                                   const GridStructure &grid,
                                   const TimeStepInfo &dt_info) const {
  const int stage = dt_info.stage;
  const auto u_stages = state->u_cf_stages();

  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  const auto uaf = state->u_af();

  const auto &order = basis_->get_order();
  static constexpr int ilo = 1;
  static const auto &ihi = grid.get_ihi();

  // --- Apply BC ---
  bc::fill_ghost_zones<3>(ucf, &grid, basis_, bcs_, {0, 2});

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "Hydro :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {ihi + 1, order, NUM_VARS_}),
      KOKKOS_LAMBDA(const int ix, const int k, const int q) {
        dU(ix, k, q) = 0.0;
      });

  // --- Fluid Increment : Divergence ---
  fluid_divergence(state, dU, grid, stage);

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Hydro :: Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ilo, 0, 0},
                                             {ihi + 1, order, NUM_VARS_}),
      KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
        dU(ix, k, q) /= basis_->get_mass_matrix(ix, k);
      });

  // --- Increment from Geometry ---
  if (grid.do_geometry()) {
    fluid_geometry(ucf, uaf, dU, grid);
  }
}

// Compute the divergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
KOKKOS_FUNCTION
void HydroPackage::fluid_divergence(const State *const state, View3D<double> dU,
                                    const GridStructure &grid,
                                    const int stage) const {
  const auto u_stages = state->u_cf_stages();

  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  const auto uaf = state->u_af();

  const auto &nNodes = grid.get_n_nodes();
  const auto &order = basis_->get_order();
  static constexpr int ilo = 1;
  static const auto &ihi = grid.get_ihi();

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  Kokkos::parallel_for(
      "Hydro :: Interface States",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 2, NUM_VARS_}),
      KOKKOS_CLASS_LAMBDA(const int ix, const int q) {
        u_f_l_(ix, q) = basis_->basis_eval(ucf, ix - 1, q, nNodes + 1);
        u_f_r_(ix, q) = basis_->basis_eval(ucf, ix, q, 0);
      });

  // --- Calc numerical flux at all faces ---
  Kokkos::parallel_for(
      "Hydro :: Numerical Fluxes", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        const double P_L = uaf(ix - 1, nNodes + 1, 0);
        const double Cs_L = uaf(ix - 1, nNodes + 1, 2);

        const double P_R = uaf(ix, 0, 0);
        const double Cs_R = uaf(ix, 0, 2);

        // --- Numerical Fluxes ---

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( u_f_l_(ix,  1 ),
        // u_f_r_(ix,  1
        // ), P_L, P_R, lam_L, lam_R);
        const auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
            u_f_l_(ix, 0), u_f_r_(ix, 0), u_f_l_(ix, 1), u_f_r_(ix, 1), P_L,
            P_R, Cs_L, Cs_R);
        flux_u_(stage, ix) = flux_u;

        dFlux_num_(ix, 0) = -flux_u_(stage, ix);
        dFlux_num_(ix, 1) = flux_p;
        dFlux_num_(ix, 2) = +flux_u_(stage, ix) * flux_p;
      });

  flux_u_(stage, ilo - 1) = flux_u_(stage, ilo);
  flux_u_(stage, ihi + 2) = flux_u_(stage, ihi + 1);

  // --- Surface Term ---
  Kokkos::parallel_for(
      "Hydro :: Surface Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ilo, 0, 0},
                                             {ihi + 1, order, NUM_VARS_}),
      KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
        const auto &Poly_L = basis_->get_phi(ix, 0, k);
        const auto &Poly_R = basis_->get_phi(ix, nNodes + 1, k);
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
        "Hydro :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
        KOKKOS_CLASS_LAMBDA(const int ix, const int k) {
          double local_sum1 = 0.0;
          double local_sum2 = 0.0;
          double local_sum3 = 0.0;
          for (int iN = 0; iN < nNodes; ++iN) {
            const double weight = grid.get_weights(iN);
            const double dphi = basis_->get_d_phi(ix, iN + 1, k);
            const double X = grid.node_coordinate(ix, iN);
            const double sqrt_gm = grid.get_sqrt_gm(X);

            const double vel = basis_->basis_eval(ucf, ix, 1, iN + 1);
            const double P = uaf(ix, iN + 1, 0);
            const auto [flux1, flux2, flux3] = flux_fluid(vel, P);

            local_sum1 += weight * flux1 * dphi * sqrt_gm;
            local_sum2 += weight * flux2 * dphi * sqrt_gm;
            local_sum3 += weight * flux3 * dphi * sqrt_gm;
          }

          dU(ix, k, 0) += local_sum1;
          dU(ix, k, 1) += local_sum2;
          dU(ix, k, 2) += local_sum3;
        });
  }
}

KOKKOS_FUNCTION
void HydroPackage::fluid_geometry(const View3D<double> ucf,
                                  const View3D<double> uaf, View3D<double> dU,
                                  const GridStructure &grid) const {
  const int &nNodes = grid.get_n_nodes();
  const int &order = basis_->get_order();
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

          local_sum +=
              grid.get_weights(iN) * P * basis_->get_phi(ix, iN + 1, k) * X;
        }

        dU(ix, k, 1) += (2.0 * local_sum * grid.get_widths(ix)) /
                        basis_->get_mass_matrix(ix, k);
      });
}
/**
 * @brief explicit hydrodynamic timestep restriction
 **/
KOKKOS_FUNCTION
auto HydroPackage::min_timestep(const State *const state,
                                const GridStructure &grid,
                                const TimeStepInfo & /*dt_info*/) const
    -> double {
  const auto ucf = state->u_cf();
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  static constexpr int ilo = 1;
  static const int &ihi = grid.get_ihi();

  double dt_out = 0.0;
  Kokkos::parallel_reduce(
      "Hydro::min_timestep", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix, double &lmin) {
        // --- Using Cell Averages ---
        const double tau_x = ucf(ix, 0, 0);
        const double vel_x = ucf(ix, 0, 1);
        const double eint_x = ucf(ix, 0, 2);

        const double dr = grid.get_widths(ix);

        // NOTE: This is not really correct. I'm using a nodal location for
        // getting the ionization terms but cell average quantities for the
        // sound speed. This is only an issue in pure hydro + ionization
        // which should be an edge case.
        // TODO(astrobarker): implement cell averaged Paczynski terms?
        double lambda[8];
        if (state->ionization_enabled()) {
          paczynski_terms(state, ix, 1, lambda);
        }
        const double Cs =
            sound_speed_from_conserved(eos_, tau_x, vel_x, eint_x, lambda);
        const double eigval = Cs + std::abs(vel_x);

        const double dt_old = std::abs(dr) / std::abs(eigval);

        lmin = std::min(dt_old, lmin);
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  return dt_out;
}

/**
 * @brief fill Hydro derived quantities for output
 *
 * TODO(astrobarker): extend
 */
void HydroPackage::fill_derived(State *const state, const GridStructure &grid,
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
  bc::fill_ghost_zones<3>(uCF, &grid, basis_, bcs_, {0, 2});

  if (state->composition_enabled()) {
    fill_derived_comps(state, &grid, basis_);
  }

  if (ionization_enabled) {
    fill_derived_ionization(state, &grid, basis_);
  }

  Kokkos::parallel_for(
      "Hydro::fill_derived", Kokkos::RangePolicy<>(ilo, ihi),
      KOKKOS_CLASS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes + 2; ++iN) {
          const double tau = basis_->basis_eval(uCF, ix, 0, iN);
          const double vel = basis_->basis_eval(uCF, ix, 1, iN);
          const double emt = basis_->basis_eval(uCF, ix, 2, iN);

          const double rho = 1.0 / tau;
          const double momentum = rho * vel;
          const double sie = (emt - 0.5 * vel * vel);

          double lambda[8];
          // This is probably not the cleanest logic, but setups with
          // ionization enabled and Paczynski disbled are an outlier.
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

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::name() const noexcept
    -> std::string_view {
  return "Hydro";
}

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::is_active() const noexcept
    -> bool {
  return active_;
}

KOKKOS_FUNCTION
void HydroPackage::set_active(const bool active) { active_ = active; }

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::get_flux_u(const int stage,
                                                            const int ix) const
    -> double {
  return flux_u_(stage, ix);
}

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::get_basis() const
    -> const ModalBasis * {
  return basis_;
}

} // namespace fluid
