#pragma once
/**
 * @file sod.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Sod shock tube
 */

#include "abstractions.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize Sod shock tube
 **/
void sod_init(State* state, GridStructure* grid, ProblemIn* pin) {

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

  const auto V_L = pin->param()->get<double>("problem.params.vL", 0.0);
  const auto V_R = pin->param()->get<double>("problem.params.vR", 0.0);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 1.0);
  const auto D_R = pin->param()->get<double>("problem.params.rhoR", 0.125);
  const auto P_L = pin->param()->get<double>("problem.params.pL", 1.0);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 0.1);
  const auto x_d = pin->param()->get<double>("problem.params.x_d", 0.5);

  const double gamma = 1.4;
  double X1          = 0.0;
  for (int iX = ilo; iX <= ihi; iX++) {
    for (int k = 0; k < pOrder; k++) {
      for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
        X1                  = grid->get_centers(iX);
        uCF(iCF_Tau, iX, k) = 0.0;
        uCF(iCF_V, iX, k)   = 0.0;
        uCF(iCF_E, iX, k)   = 0.0;

        if (X1 <= x_d) {
          if (k == 0) {
            uCF(iCF_Tau, iX, 0) = 1.0 / D_L;
            uCF(iCF_V, iX, 0)   = V_L;
            uCF(iCF_E, iX, 0) =
                (P_L / (gamma - 1.0)) * uCF(iCF_Tau, iX, 0) + 0.5 * V_L * V_L;
          }

          uPF(iPF_D, iX, iNodeX) = D_L;
        } else { // right domain
          if (k == 0) {
            uCF(iCF_Tau, iX, 0) = 1.0 / D_R;
            uCF(iCF_V, iX, 0)   = V_R;
            uCF(iCF_E, iX, 0) =
                (P_R / (gamma - 1.0)) * uCF(iCF_Tau, iX, 0) + 0.5 * V_R * V_R;
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
