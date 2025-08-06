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
void smooth_flow_init(State* state, GridStructure* grid, ProblemIn* pin) {

  View3D<double> uCF = state->get_u_cf();
  View3D<double> uPF = state->get_u_pf();

  const int ilo    = grid->get_ilo();
  const int ihi    = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  const auto amp =
      pin->param()->get<double>("problem.params.amp", 0.9999999999999999);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(int iX) {
        const int k = 0;
        const double X1 = grid->get_centers(iX);

        const double D = (1.0 + (amp * sin(constants::PI * X1)));
        uCF(iCF_Tau, iX, k) = 1.0 / D;
        uCF(iCF_V, iX, k)   = 0.0;
        uCF(iCF_E, iX, k)   = (D * D * D / 2.0) * uCF(iCF_Tau, iX, k);

        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(iPF_D, iX, iNodeX) = (1.0 + amp * sin(constants::PI * X1));
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
