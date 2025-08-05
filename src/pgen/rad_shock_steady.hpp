#pragma once
/**
 * @file rad_equilibrium.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation fluid equilibriation test
 */

#include <cmath> /* sin */

#include "abstractions.hpp"
#include "constants.hpp"
#include "grid.hpp"
#include "state.hpp"

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
void rad_shock_steady_init(State* state, GridStructure* grid, ProblemIn* pin) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  if (!rad_active) {
    THROW_ATHELAS_ERROR("Steady radiative shock requires radiation enabled!");
  }

  View3D<double> uCF = state->get_u_cf();
  View3D<double> uPF = state->get_u_pf();
  const int pOrder   = state->get_p_order();

  const int ilo    = grid->get_ilo();
  const int ihi    = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  const int iCF_Tau = 0;
  const int iCF_V   = 1;
  const int iCF_E   = 2;

  const int iPF_D = 0;

  const int iCR_E = 3;

  const auto V0   = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto rhoL = pin->param()->get<double>("problem.params.rhoL", 1.0);
  const auto rhoR = pin->param()->get<double>("problem.params.rhoR", 2.286);
  const auto T_L =
      pin->param()->get<double>("problem.params.T_L", 1.16045181e6); // K
  const auto T_R =
      pin->param()->get<double>("problem.params.T_R", 2.4109e6); // K

  // TODO(astrobarker): thread through
  const double Abar  = 1.0;
  const double gamma = 5.0 / 3.0;
  const double em_gas_L =
      (T_L * constants::N_A * constants::k_B) / ((gamma - 1.0) * Abar);
  const double em_gas_R =
      (T_R * constants::N_A * constants::k_B) / ((gamma - 1.0) * Abar);
  const double e_rad_L = constants::a * std::pow(T_L, 4.0);
  const double e_rad_R = constants::a * std::pow(T_R, 4.0);

  for (int iX = 0; iX <= ihi + 1; iX++) {
    for (int k = 0; k < pOrder; k++) {
      for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
        double X1           = grid->get_centers(iX);
        uCF(iCF_Tau, iX, k) = 0.0;
        uCF(iCF_V, iX, k)   = 0.0;
        uCF(iCF_E, iX, k)   = 0.0;
        uCF(3, iX, k)       = 0.0;
        uCF(4, iX, k)       = 0.0;

        if (X1 <= 0.0) {
          if (k == 0) {
            uCF(iCF_Tau, iX, 0) = 1.0 / rhoL;
            uCF(iCF_V, iX, 0)   = V0;
            uCF(iCF_E, iX, 0)   = em_gas_L;

            uCF(iCR_E, iX, 0) = e_rad_L;
          }
          uPF(iPF_D, iX, iNodeX) = rhoL;
        } else {
          if (k == 0) {
            uCF(iCF_Tau, iX, 0) = 1.0 / rhoR;
            uCF(iCF_V, iX, 0)   = V0;
            uCF(iCF_E, iX, 0)   = em_gas_R;

            uCF(iCR_E, iX, 0) = e_rad_R;
          }
          uPF(iPF_D, iX, iNodeX) = rhoR;
        }
      }
    }
  }
  // Fill density in guard cells
  for (int iX = 0; iX < ilo; iX++) {
    for (int iN = 0; iN < nNodes; iN++) {
      uPF(0, ilo - 1 - iX, iN) = uPF(0, ilo + iX, nNodes - iN - 1);
      uPF(0, ihi + 1 + iX, iN) = uPF(0, ihi - iX, nNodes - iN - 1);
    }
  }
}
