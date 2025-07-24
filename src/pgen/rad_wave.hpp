#pragma once
/**
 * @file rad_shock.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation shock test
 */

#include "abstractions.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize radiating shock
 **/
void rad_shock_init(State* state, GridStructure* grid, const ProblemIn* pin) {
  View3D<double> uCF = state->get_u_cf();
  View3D<double> uPF = state->get_u_pf();
  const int pOrder   = state->get_p_order();

  const int ilo    = grid->get_ilo();
  const int ihi    = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  constexpr static int iCR_E = 3;

  const double lambda =
      pin->in_table["problem"]["params"]["lambda"].value_or(0.1);
  const double kappa =
      pin->in_table["problem"]["params"]["kappa"].value_or(1.0);
  const double epsilon =
      pin->in_table["problem"]["params"]["epsilon"].value_or(1.0e-6);
  const double rho0 = pin->in_table["problem"]["params"]["rho0"].value_or(1.0);
  const double P0 =
      pin->in_table["problem"]["params"]["p0"].value_or(1.0e-6); // K

  // TODO(astrobarker): thread through
  const double gamma = 5.0 / 3.0;
  const double gm1   = gamma - 1.0;

  for (int iX = 0; iX <= ihi + 1; iX++) {
    for (int k = 0; k < pOrder; k++) {
      for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
        double X1           = grid->get_centers(iX);
        uCF(iCF_Tau, iX, k) = 0.0;
        uCF(iCF_V, iX, k)   = 0.0;
        uCF(iCF_E, iX, k)   = 0.0;
        uCF(0, iX, k)       = 0.0;
        uCF(1, iX, k)       = 0.0;

        if (k == 0) {
          uCF(iCF_Tau, iX, 0) = 1.0 / rho0;
          uCF(iCF_V, iX, 0)   = 0.0;
          uCF(iCF_E, iX, 0)   = em_gas_R + 0.5 * V_R * V_R;

          uCF(iCR_E, iX, 0) = em_rad_R;
        }
        uPF(iPF_D, iX, iNodeX) = rho0;
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
