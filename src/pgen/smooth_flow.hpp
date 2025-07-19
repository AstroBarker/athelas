#pragma once
/**
 * @file smooth_flow.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Smooth flow test
 */

#include <cmath>

#include "abstractions.hpp"
#include "constants.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize smooth flow test problem
 **/
void smooth_flow_init(State* state, GridStructure* grid, const ProblemIn* pin) {

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

  const double amp =
      pin->in_table["problem"]["params"]["amp"].value_or(0.9999999999999999999);

  double X1 = 0.0;
  for (int iX = ilo; iX <= ihi; iX++) {
    for (int k = 0; k < pOrder; k++) {
      for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
        X1                  = grid->get_centers(iX);
        uCF(iCF_Tau, iX, k) = 0.0;
        uCF(iCF_V, iX, k)   = 0.0;
        uCF(iCF_E, iX, k)   = 0.0;

        if (k == 0) {
          double const D      = (1.0 + (amp * sin(constants::PI * X1)));
          uCF(iCF_Tau, iX, 0) = 1.0 / D;
          uCF(iCF_V, iX, 0)   = 0.0;
          uCF(iCF_E, iX, 0)   = (D * D * D / 2.0) * uCF(iCF_Tau, iX, 0);
        } else if (k == 1) {
          double const D      = (1.0 + (amp * sin(constants::PI * X1)));
          double const dD     = (amp * constants::PI * cos(constants::PI * X1));
          uCF(iCF_Tau, iX, k) = (-1 / (D * D)) * dD * grid->get_widths(iX);
          uCF(iCF_V, iX, k)   = 0.0;
          uCF(iCF_E, iX, k)   = ((2.0 / 2.0) * D) * dD * grid->get_widths(iX);
        } else if (k == 2) {
          double const D = (1.0 + (amp * sin(constants::PI * X1)));
          double const ddD =
              -(amp * constants::PI * constants::PI) * sin(constants::PI * X1);
          uCF(iCF_Tau, iX, k) = (2.0 / (D * D * D)) * ddD *
                                grid->get_widths(iX) * grid->get_widths(iX);
          uCF(iCF_V, iX, k) = 0.0;
          uCF(iCF_E, iX, k) =
              (2.0 / 2.0) * ddD * grid->get_widths(iX) * grid->get_widths(iX);
        } else if (k == 3) {
          double const D = (1.0 + (amp * sin(constants::PI * X1)));
          double const dddD =
              -(amp * constants::PI * constants::PI * constants::PI) *
              cos(constants::PI * X1);
          uCF(iCF_Tau, iX, k) = (-6.0 / (D * D * D * D)) * dddD *
                                grid->get_widths(iX) * grid->get_widths(iX) *
                                grid->get_widths(iX);
          uCF(iCF_V, iX, k) = 0.0;
          uCF(iCF_E, iX, k) = 0.0;
        } else {
          uCF(iCF_Tau, iX, k) = 0.0;
          uCF(iCF_V, iX, k)   = 0.0;
          uCF(iCF_E, iX, k)   = 0.0;
        }

        uPF(iPF_D, iX, iNodeX) = (1.0 + amp * sin(constants::PI * X1));
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
