#pragma once
/**
 * @file advection.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Fluid advection test
 */

#include <cmath> /* sin */

#include "abstractions.hpp"
#include "constants.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * Initialize advection test
 **/
void advection_init(State* state, GridStructure* grid, const ProblemIn* pin) {
  // Smooth advection problem
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

  const double V0  = pin->in_table["problem"]["params"]["v0"].value_or(-1.0);
  const double P0  = pin->in_table["problem"]["params"]["p0"].value_or(0.01);
  const double Amp = pin->in_table["problem"]["params"]["amp"].value_or(1.0);

  double X1 = 0.0;
  for (int iX = ilo; iX <= ihi; iX++) {
    for (int k = 0; k < pOrder; k++) {
      for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
        X1 = grid->get_centers(iX);

        if (k != 0) {
          uCF(iCF_Tau, iX, k) = 0.0;
          uCF(iCF_V, iX, k)   = 0.0;
          uCF(iCF_E, iX, k)   = 0.0;
        } else {
          uCF(iCF_Tau, iX, k) =
              1.0 / (2.0 + Amp * sin(2.0 * constants::PI * X1));
          uCF(iCF_V, iX, k) = V0;
          uCF(iCF_E, iX, k) = (P0 / 0.4) * uCF(iCF_Tau, iX, k) + 0.5 * V0 * V0;
        }
        uPF(iPF_D, iX, iNodeX) = (2.0 + Amp * sin(2.0 * constants::PI * X1));
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
