#pragma once
/**
 * @file sedov.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Sedov blast wave
 */

#include <cmath>

#include "abstractions.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize sedov blast wave
 **/
void sedov_init(State* state, GridStructure* grid, ProblemIn* pin) {

  View3D<double> uCF = state->get_u_cf();
  View3D<double> uPF = state->get_u_pf();

  const int ilo    = grid->get_ilo();
  const int ihi    = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  const auto D0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto E0 = pin->param()->get<double>("problem.params.E0", 0.3);

  const int origin = 1;

  // TODO(astrobarker): geometry aware volume for energy
  const double volume =
      (4.0 * M_PI / 3.0) * std::pow(grid->get_left_interface(origin + 1), 3.0);
  const double gamma = 1.4;
  const double P0    = (gamma - 1.0) * E0 / volume;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(int iX) {
        const int k = 0;
        
        uCF(iCF_Tau, iX, k) = 1.0 / D0;
        uCF(iCF_V, iX, k)   = V0;
        if (iX == origin - 1 || iX == origin) {
          uCF(iCF_E, iX, k) =
              (P0 / (gamma - 1.0)) * uCF(iCF_Tau, iX, k) + 0.5 * V0 * V0;
        } else {
          uCF(iCF_E, iX, k) =
              (1.0e-6 / (gamma - 1.0)) * uCF(iCF_Tau, iX, k) + 0.5 * V0 * V0;
        }
        
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(iPF_D, iX, iNodeX) = D0;
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
