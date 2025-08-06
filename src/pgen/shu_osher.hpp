#pragma once
/**
 * @file shu_osher.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Shu Osher shock tube
 */

#include <cmath>

#include "abstractions.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize Shu Osher hydro test
 **/
void shu_osher_init(State* state, GridStructure* grid, ProblemIn* pin) {

  View3D<double> uCF = state->get_u_cf();
  View3D<double> uPF = state->get_u_pf();

  const int ilo    = grid->get_ilo();
  const int ihi    = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  const auto V0  = pin->param()->get<double>("problem.params.v0", 2.629369);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 3.857143);
  const auto P_L =
      pin->param()->get<double>("problem.params.pL", 10.333333333333);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 1.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 1),
      KOKKOS_LAMBDA(int iX) {
        const int k = 0;
        const double X1 = grid->get_centers(iX);

        if (X1 <= -4.0) {
          uCF(iCF_Tau, iX, k) = 1.0 / D_L;
          uCF(iCF_V, iX, k)   = V0;
          uCF(iCF_E, iX, k) =
              (P_L / 0.4) * uCF(iCF_Tau, iX, k) + 0.5 * V0 * V0;

          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(iPF_D, iX, iNodeX) = D_L;
          }
        } else { // right domain
          uCF(iCF_Tau, iX, k) = 1.0 / (1.0 + 0.2 * sin(5.0 * X1));
          uCF(iCF_V, iX, k)   = 0.0;
          uCF(iCF_E, iX, k)   = (P_R / 0.4) * uCF(iCF_Tau, iX, k);

          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(iPF_D, iX, iNodeX) = (1.0 + 0.2 * sin(5.0 * X1));
          }
        }
      });

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo),
      KOKKOS_LAMBDA(int iX) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(0, ilo - 1 - iX, iN) = uPF(0, ilo + iX, nNodes - iN - 1);
          uPF(0, ihi + 1 + iX, iN) = uPF(0, ihi - iX, nNodes - iN - 1);
        }
      });
}
