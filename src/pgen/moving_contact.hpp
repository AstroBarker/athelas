/**
 * @file moving_contact.hpp
 * --------------
 *
 * @brief Moving contact wave test
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

namespace athelas {

/**
 * @brief Initialize moving contact discontinuity test
 **/
void moving_contact_init(State *state, GridStructure *grid, ProblemIn *pin,
                         const eos::EOS *eos,
                         basis::ModalBasis * /*fluid_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Moving contact requires ideal gas eos!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();

  const int ilo = 1;
  const int ihi = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.1);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 1.4);
  const auto D_R = pin->param()->get<double>("problem.params.rhoR", 1.0);
  const auto P_L = pin->param()->get<double>("problem.params.pL", 1.0);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 1.0);

  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        const int k = 0;
        const double X1 = grid->centers(ix);

        if (X1 <= 0.5) {
          uCF(ix, k, q_Tau) = 1.0 / D_L;
          uCF(ix, k, q_V) = V0;
          uCF(ix, k, q_E) = (P_L / gm1) * uCF(ix, k, q_Tau) + 0.5 * V0 * V0;

          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(ix, iNodeX, iPF_D) = D_L;
          }
        } else {
          uCF(ix, k, q_Tau) = 1.0 / D_R;
          uCF(ix, k, q_V) = V0;
          uCF(ix, k, q_E) = (P_R / gm1) * uCF(ix, k, q_Tau) + 0.5 * V0 * V0;

          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(ix, iNodeX, iPF_D) = D_R;
          }
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

} // namespace athelas
