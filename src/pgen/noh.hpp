/**
 * @file noh.hpp
 * --------------
 *
 * @brief Noh test
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize Noh problem
 **/
void noh_init(State *state, GridStructure *grid, ProblemIn *pin,
              const eos::EOS *eos,
              basis::ModalBasis * /*fluid_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Noh requires ideal gas eos!");
  }

  AthelasArray3D<double> uCF = state->u_cf();
  AthelasArray3D<double> uPF = state->u_pf();

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->get_n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto P0 = pin->param()->get<double>("problem.params.p0", 0.000001);
  const auto V0 = pin->param()->get<double>("problem.params.v0", -1.0);
  const auto D0 = pin->param()->get<double>("problem.params.rho0", 1.0);

  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Noh (1)", DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i) {
        const int k = 0;

        uCF(i, k, q_Tau) = 1.0 / D0;
        uCF(i, k, q_V) = V0;
        uCF(i, k, q_E) = (P0 / gm1) * uCF(i, k, q_Tau) + 0.5 * V0 * V0;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(i, iNodeX, iPF_D) = D0;
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Noh (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
