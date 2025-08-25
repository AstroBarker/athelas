#pragma once
/**
 * @file shockless_noh.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Shockless Noh collapse
 */

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * @brief Initialize shockless Noh problem
 **/
void shockless_noh_init(State* state, GridStructure* grid, ProblemIn* pin,
                        const EOS* /*eos*/,
                        ModalBasis* /*fluid_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Shockless Noh requires ideal gas eos!");
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

  const auto D = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto E_M =
      pin->param()->get<double>("problem.params.specific_energy", 1.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        const int k = 0;
        const double X1 = grid->get_centers(ix);

        uCF(ix, k, q_Tau) = 1.0 / D;
        uCF(ix, k, q_V) = -X1;
        uCF(ix, k, q_E) = E_M + 0.5 * uCF(ix, k, q_V) * uCF(ix, k, q_V);

        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(ix, iNodeX, iPF_D) = D;
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
