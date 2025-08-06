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
#include "polynomial_basis.hpp"

/**
 * @brief Initialize Shu Osher hydro test
 **/
void shu_osher_init(State* state, GridStructure* grid, ProblemIn* pin,
                     ModalBasis* fluid_basis = nullptr) {

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

  // Phase 1: Initialize nodal values (always done)
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(int iX) {
        const double X1 = grid->get_centers(iX);

        if (X1 <= -4.0) {
          // Left state: constant values
          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(iPF_D, iX, iNodeX) = D_L;
          }
        } else {
          // Right state: sinusoidal density
          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            const double x = grid->node_coordinate(iX, iNodeX);
            uPF(iPF_D, iX, iNodeX) = (1.0 + 0.2 * sin(5.0 * x));
          }
        }
      });

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    // Use L2 projection for accurate modal coefficients
    auto tau_func = [&D_L](double x) -> double {
      if (x <= -4.0) {
        return 1.0 / D_L;
      } else {
        return 1.0 / (1.0 + 0.2 * sin(5.0 * x));
      }
    };
    
    auto velocity_func = [&V0](double x) -> double {
      if (x <= -4.0) {
        return V0;
      } else {
        return 0.0;
      }
    };
    
    auto energy_func = [&P_L, &P_R, &V0, &D_L](double x) -> double {
      if (x <= -4.0) {
        return (P_L / 0.4) / D_L + 0.5 * V0 * V0;
      } else {
        const double rho = 1.0 + 0.2 * sin(5.0 * x);
        return (P_R / 0.4) / rho;
      }
    };
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(ilo, ihi + 1),
        KOKKOS_LAMBDA(int iX) {
          const int k = 0;
          const double X1 = grid->get_centers(iX);

          if (X1 <= -4.0) {
            uCF(iCF_Tau, iX, k) = 1.0 / D_L;
            uCF(iCF_V, iX, k)   = V0;
            uCF(iCF_E, iX, k) =
                (P_L / 0.4) * uCF(iCF_Tau, iX, k) + 0.5 * V0 * V0;
          } else {
	    // Project each conserved variable
	    fluid_basis->project_nodal_to_modal(uCF, uPF, grid, iCF_Tau, iX, tau_func);
	    fluid_basis->project_nodal_to_modal(uCF, uPF, grid, iCF_V, iX, velocity_func);
	    fluid_basis->project_nodal_to_modal(uCF, uPF, grid, iCF_E, iX, energy_func);
          }
        });
    
  } else {
    // Fallback: set cell averages only (k=0)
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(ilo, ihi + 1),
        KOKKOS_LAMBDA(int iX) {
          const int k = 0;
          const double X1 = grid->get_centers(iX);

          if (X1 <= -4.0) {
            uCF(iCF_Tau, iX, k) = 1.0 / D_L;
            uCF(iCF_V, iX, k)   = V0;
            uCF(iCF_E, iX, k) =
                (P_L / 0.4) * uCF(iCF_Tau, iX, k) + 0.5 * V0 * V0;
          } else {
            uCF(iCF_Tau, iX, k) = 1.0 / (1.0 + 0.2 * sin(5.0 * X1));
            uCF(iCF_V, iX, k)   = 0.0;
            uCF(iCF_E, iX, k)   = (P_R / 0.4) * uCF(iCF_Tau, iX, k);
          }
        });
  }

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
