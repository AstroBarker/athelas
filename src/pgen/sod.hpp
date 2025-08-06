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

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(int iX) {
        const int k = 0;
        const double X1 = grid->get_centers(iX);

        if (X1 <= x_d) {
          uCF(iCF_Tau, iX, k) = 1.0 / D_L;
          uCF(iCF_V, iX, k)   = V_L;
          uCF(iCF_E, iX, k) =
              (P_L / (gamma - 1.0)) * uCF(iCF_Tau, iX, k) + 0.5 * V_L * V_L;

          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(iPF_D, iX, iNodeX) = D_L;
          }
        } else { // right domain
          uCF(iCF_Tau, iX, k) = 1.0 / D_R;
          uCF(iCF_V, iX, k)   = V_R;
          uCF(iCF_E, iX, k) =
              (P_R / (gamma - 1.0)) * uCF(iCF_Tau, iX, k) + 0.5 * V_R * V_R;

          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(iPF_D, iX, iNodeX) = D_R;
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
