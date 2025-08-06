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
void advection_init(State* state, GridStructure* grid, ProblemIn* pin,
                    ModalBasis* fluid_basis = nullptr) {
  // Smooth advection problem
  View3D<double> uCF = state->get_u_cf();
  View3D<double> uPF = state->get_u_pf();

  const int ilo    = grid->get_ilo();
  const int ihi    = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  const int iCF_Tau = 0;
  const int iCF_V   = 1;
  const int iCF_E   = 2;

  const int iPF_D = 0;

  const auto V0  = pin->param()->get<double>("problem.params.v0", -1.0);
  const auto P0  = pin->param()->get<double>("problem.params.p0", 0.01);
  const auto Amp = pin->param()->get<double>("problem.params.amp", 1.0);

  // Phase 1: Initialize nodal values (always done)
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(int iX) {
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          const double x = grid->node_coordinate(iX, iNodeX);
          uPF(iPF_D, iX, iNodeX) = (2.0 + Amp * sin(2.0 * constants::PI * x));
        }
      });

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    // Use L2 projection for accurate modal coefficients
    auto density_func = [&Amp](double x) -> double {
      return 2.0 + Amp * sin(2.0 * constants::PI * x);
    };
    
    auto velocity_func = [&V0](double x) -> double {
      return V0;
    };
    
    auto energy_func = [&P0, &V0, &Amp](double x) -> double {
      const double rho = 2.0 + Amp * sin(2.0 * constants::PI * x);
      return (P0 / 0.4) / rho + 0.5 * V0 * V0;
    };
    
    // Project each conserved variable using Kokkos parallel for
    fluid_basis->project_nodal_to_modal_all_cells(uCF, uPF, grid, iCF_Tau, density_func);
    fluid_basis->project_nodal_to_modal_all_cells(uCF, uPF, grid, iCF_V, velocity_func);
    fluid_basis->project_nodal_to_modal_all_cells(uCF, uPF, grid, iCF_E, energy_func);
  } else {
    // Fallback: set cell averages only (k=0)
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(ilo, ihi + 1),
        KOKKOS_LAMBDA(int iX) {
          const int k = 0;
          const double X1 = grid->get_centers(iX);

          uCF(iCF_Tau, iX, k) =
              1.0 / (2.0 + Amp * sin(2.0 * constants::PI * X1));
          uCF(iCF_V, iX, k) = V0;
          uCF(iCF_E, iX, k) = (P0 / 0.4) * uCF(iCF_Tau, iX, k) + 0.5 * V0 * V0;
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
