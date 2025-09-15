#pragma once
/**
 * @file rad_equilibrium.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation fluid equilibriation test
 */

#include <cmath> /* sin */

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"
#include "utils/constants.hpp"

/**
 * @brief Initialize steady radiating shock
 *
 * Two different cases: Mach 2 and Mach 5.
 *
 * Mach 2 Case:
 * - Left side (pre-shock):
 *   - Density: 1.0 g/cm^3
 *   - Temperature: 1.16045181e6 K (100eV)
 * - Right side (post-shock):
 *   - Density: 2.286 g/cm^3
 *   - Temperature: 2.4109e6 K (207.756 eV)
 *
 * Mach 5 Case:
 * - Left side (pre-shock):
 *   - Density: 1.0 g/cm^3
 *   - Temperature: 1.16045181e6 K (100 eV)
 * - Right side (post-shock):
 *   - Density: 3.598 g/cm^3
 *   - Temperature: 9.9302e6 K (855.720 eV)
 **/
void rad_shock_steady_init(State *state, GridStructure *grid, ProblemIn *pin,
                           const EOS *eos,
                           ModalBasis * /*fluid_basis = nullptr*/,
                           ModalBasis * /*radiation_basis = nullptr*/) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  if (!rad_active) {
    THROW_ATHELAS_ERROR("Steady radiative shock requires radiation enabled!");
  }

  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Steady radiative shock requires ideal gas eos!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();

  static const int ilo = 1;
  static const int ihi = grid->get_ihi();
  static const int nNodes = grid->get_n_nodes();

  const int q_Tau = 0;
  const int q_V = 1;
  const int q_E = 2;

  const int iPF_D = 0;

  const int iCR_E = 3;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto rhoL = pin->param()->get<double>("problem.params.rhoL", 1.0);
  const auto rhoR = pin->param()->get<double>("problem.params.rhoR", 2.286);
  const auto T_L =
      pin->param()->get<double>("problem.params.T_L", 1.16045181e6); // K
  const auto T_R =
      pin->param()->get<double>("problem.params.T_R", 2.4109e6); // K

  // TODO(astrobarker): thread through
  const double Abar = 1.0;
  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;
  const double em_gas_L =
      (T_L * constants::N_A * constants::k_B) / (gm1 * Abar);
  const double em_gas_R =
      (T_R * constants::N_A * constants::k_B) / (gm1 * Abar);
  const double e_rad_L = constants::a * std::pow(T_L, 4.0);
  const double e_rad_R = constants::a * std::pow(T_R, 4.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 2), KOKKOS_LAMBDA(int ix) {
        const int k = 0;
        const double X1 = grid->get_centers(ix);

        if (X1 <= 0.0) {
          uCF(ix, k, q_Tau) = 1.0 / rhoL;
          uCF(ix, k, q_V) = V0;
          uCF(ix, k, q_E) = em_gas_L;
          uCF(ix, k, iCR_E) = e_rad_L;

          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(ix, iNodeX, iPF_D) = rhoL;
          }
        } else {
          uCF(ix, k, q_Tau) = 1.0 / rhoR;
          uCF(ix, k, q_V) = V0;
          uCF(ix, k, q_E) = em_gas_R;
          uCF(ix, k, iCR_E) = e_rad_R;

          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(ix, iNodeX, iPF_D) = rhoR;
          }
        }
      });

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(ilo - 1 - ix, iN, 0) = uPF(ilo + ix, nNodes - iN - 1, 0);
          uPF(ilo + 1 + ix, iN, 0) = uPF(ilo - ix, nNodes - iN - 1, 0);
        }
      });
}
