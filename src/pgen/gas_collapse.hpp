/**
 * @file gas_collapse.hpp
 * --------------
 *
 * @brief Collapsing gas cloud
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

namespace athelas {

/**
 * @brief Initialize gas collapse
 **/
void gas_collapse_init(State *state, GridStructure *grid, ProblemIn *pin,
                       const eos::EOS *eos,
                       basis::ModalBasis * /*fluid_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Gas collapse requires ideal gas eos!");
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

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto p0 = pin->param()->get<double>("problem.params.p0", 10.0);

  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        const int k = 0;

        uCF(ix, k, q_Tau) = rho0; // / rho0 * (1.0 / std::cosh(x / H));
        uCF(ix, k, q_V) = V0;
        uCF(ix, k, q_E) = (p0 / gm1) * uCF(ix, k, q_Tau) + 0.5 * V0 * V0;

        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(ix, iNodeX, iPF_D) = rho0;
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
