#pragma once
/**
 * @file shu_osher.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Shu Osher shock tube
 */

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * @brief Initialize Shu Osher hydro test
 **/
void shu_osher_init(State *state, GridStructure *grid, ProblemIn *pin,
                    const EOS *eos, ModalBasis *fluid_basis = nullptr) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Shu Osher requires ideal gas eos!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();

  static const int ilo = 1;
  static const int ihi = grid->get_ihi();
  static const int nNodes = grid->get_n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 2.629369);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 3.857143);
  const auto P_L =
      pin->param()->get<double>("problem.params.pL", 10.333333333333);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 1.0);

  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;

  // Phase 1: Initialize nodal values (always done)
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        const double X1 = grid->get_centers(ix);

        if (X1 <= -4.0) {
          // Left state: constant values
          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(ix, iNodeX, iPF_D) = D_L;
          }
        } else {
          // Right state: sinusoidal density
          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            const double x = grid->node_coordinate(ix, iNodeX);
            uPF(ix, iNodeX, iPF_D) = (1.0 + 0.2 * sin(5.0 * x));
          }
        }
      });

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    // Use L2 projection for accurate modal coefficients
    auto tau_func = [&D_L](double x, int /*ix*/, int /*iN*/) -> double {
      if (x <= -4.0) {
        return 1.0 / D_L;
      }
      return 1.0 / (1.0 + 0.2 * sin(5.0 * x));
    };

    auto velocity_func = [&V0](double x, int /*ix*/, int /*iN*/) -> double {
      if (x <= -4.0) {
        return V0;
      }
      return 0.0;
    };

    auto energy_func = [&P_L, &P_R, &V0, &D_L, &gm1](double x, int /*ix*/,
                                                     int /*iN*/) -> double {
      if (x <= -4.0) {
        return (P_L / gm1) / D_L + 0.5 * V0 * V0;
      }
      const double rho = 1.0 + 0.2 * sin(5.0 * x);
      return (P_R / gm1) / rho;
    };

    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
          const int k = 0;
          const double X1 = grid->get_centers(ix);

          if (X1 <= -4.0) {
            uCF(ix, k, q_Tau) = 1.0 / D_L;
            uCF(ix, k, q_V) = V0;
            uCF(ix, k, q_E) = (P_L / gm1) * uCF(ix, k, q_Tau) + 0.5 * V0 * V0;
          } else {
            // Project each conserved variable
            fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_Tau, ix,
                                                tau_func);
            fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_V, ix,
                                                velocity_func);
            fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_E, ix,
                                                energy_func);
          }
        });

  } else {
    // Fallback: set cell averages only (k=0)
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
          const int k = 0;
          const double X1 = grid->get_centers(ix);

          if (X1 <= -4.0) {
            uCF(ix, k, q_Tau) = 1.0 / D_L;
            uCF(ix, k, q_V) = V0;
            uCF(ix, k, q_E) = (P_L / gm1) * uCF(ix, k, q_Tau) + 0.5 * V0 * V0;
          } else {
            uCF(ix, k, q_Tau) = 1.0 / (1.0 + 0.2 * sin(5.0 * X1));
            uCF(ix, k, q_V) = 0.0;
            uCF(ix, k, q_E) = (P_R / gm1) * uCF(ix, k, q_Tau);
          }
        });
  }

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(ilo - 1 - ix, iN, 0) = uPF(ilo + ix, nNodes - iN - 1, 0);
          uPF(ilo + 1 + ix, iN, 0) = uPF(ilo - ix, nNodes - iN - 1, 0);
        }
      });
}
