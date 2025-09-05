#pragma once
/**
 * @file sod.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Sod shock tube
 */

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * @brief Initialize Sod shock tube
 **/
void sod_init(State* state, GridStructure* grid, ProblemIn* pin, const EOS* eos,
              ModalBasis* /*fluid_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Sod requires ideal gas eos!");
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

  const auto V_L = pin->param()->get<double>("problem.params.vL", 0.0);
  const auto V_R = pin->param()->get<double>("problem.params.vR", 0.0);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 1.0);
  const auto D_R = pin->param()->get<double>("problem.params.rhoR", 0.125);
  const auto P_L = pin->param()->get<double>("problem.params.pL", 1.0);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 0.1);
  const auto x_d = pin->param()->get<double>("problem.params.x_d", 0.5);

  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;

  // Phase 1: Initialize nodal values (always done)
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        const double X1 = grid->get_centers(ix);

        if (X1 <= x_d) {
          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(ix, iNodeX, iPF_D) = D_L;
          }
        } else {
          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(ix, iNodeX, iPF_D) = D_R;
          }
        }
      });

  // Phase 2: Initialize modal coefficients
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        const int k = 0;
        const double X1 = grid->get_centers(ix);

        if (X1 <= x_d) {
          uCF(ix, k, q_Tau) = 1.0 / D_L;
          uCF(ix, k, q_V) = V_L;
          uCF(ix, k, q_E) = (P_L / gm1) * uCF(ix, k, q_Tau) + 0.5 * V_L * V_L;
        } else {
          uCF(ix, k, q_Tau) = 1.0 / D_R;
          uCF(ix, k, q_V) = V_R;
          uCF(ix, k, q_E) = (P_R / gm1) * uCF(ix, k, q_Tau) + 0.5 * V_R * V_R;
        }
      });

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(ilo - 1 - ix, iN, 0) = uPF(ilo + ix, nNodes - iN - 1, 0);
          uPF(ilo + 1 + ix, iN, 0) = uPF(ilo - ix, nNodes - iN - 1, 0);
        }
      });
}
