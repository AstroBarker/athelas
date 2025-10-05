/**
 * @file sedov.hpp
 * --------------
 *
 * @brief Sedov blast wave
 */

#pragma once

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

namespace athelas {

/**
 * @brief Initialize sedov blast wave
 **/
void sedov_init(State *state, GridStructure *grid, ProblemIn *pin,
                const eos::EOS *eos,
                basis::ModalBasis * /*fluid_basis = nullptr*/) {

  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Sedov requires ideal gas eos!");
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

  const auto D0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto E0 = pin->param()->get<double>("problem.params.E0", 0.3);

  const int origin = 1;

  // TODO(astrobarker): geometry aware volume for energy
  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;
  const double volume =
      (4.0 * M_PI / 3.0) * std::pow(grid->get_left_interface(origin + 1), 3.0);
  const double P0 = gm1 * E0 / volume;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        const int k = 0;

        uCF(ix, k, q_Tau) = 1.0 / D0;
        uCF(ix, k, q_V) = V0;
        if (ix == origin - 1 || ix == origin) {
          uCF(ix, k, q_E) = (P0 / gm1) * uCF(ix, k, q_Tau) + 0.5 * V0 * V0;
        } else {
          uCF(ix, k, q_E) = (1.0e-6 / gm1) * uCF(ix, k, q_Tau) + 0.5 * V0 * V0;
        }

        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(ix, iNodeX, iPF_D) = D0;
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
