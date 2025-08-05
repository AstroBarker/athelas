#pragma once
/**
 * @file moving_contact.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Moving contact wave test
 */

#include "abstractions.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize moving contact discontinuity test
 **/
void moving_contact_init(State* state, GridStructure* grid, ProblemIn* pin) {

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

  const auto V0  = pin->param()->get<double>("problem.params.v0", 0.1);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 1.4);
  const auto D_R = pin->param()->get<double>("problem.params.rhoR", 1.0);
  const auto P_L = pin->param()->get<double>("problem.params.pL", 1.0);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 1.0);

  double X1 = 0.0;
  for (int iX = ilo; iX <= ihi; iX++) {
    for (int k = 0; k < pOrder; k++) {
      for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
        X1                  = grid->get_centers(iX);
        uCF(iCF_Tau, iX, k) = 0.0;
        uCF(iCF_V, iX, k)   = 0.0;
        uCF(iCF_E, iX, k)   = 0.0;

        if (X1 <= 0.5) {
          if (k == 0) {
            uCF(iCF_Tau, iX, 0) = 1.0 / D_L;
            uCF(iCF_V, iX, 0)   = V0;
            uCF(iCF_E, iX, 0) =
                (P_L / 0.4) * uCF(iCF_Tau, iX, 0) + 0.5 * V0 * V0;
          }

          uPF(iPF_D, iX, iNodeX) = D_L;
        } else {
          if (k == 0) {
            uCF(iCF_Tau, iX, k) = 1.0 / D_R;
            uCF(iCF_V, iX, k)   = V0;
            uCF(iCF_E, iX, k) =
                (P_R / 0.4) * uCF(iCF_Tau, iX, k) + 0.5 * V0 * V0;
          }

          uPF(iPF_D, iX, iNodeX) = D_R;
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
