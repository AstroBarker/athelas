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

  View3D<double> uCF = state->get_u_cf();
  View3D<double> uPF = state->get_u_pf();

  const int ilo    = 1;
  const int ihi    = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  const auto D = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto E_M =
      pin->param()->get<double>("problem.params.specific_energy", 1.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int iX) {
        const int k     = 0;
        const double X1 = grid->get_centers(iX);

        uCF(iCF_Tau, iX, k) = 1.0 / D;
        uCF(iCF_V, iX, k)   = -X1;
        uCF(iCF_E, iX, k)   = E_M + 0.5 * uCF(iCF_V, iX, k) * uCF(iCF_V, iX, k);

        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(iPF_D, iX, iNodeX) = D;
        }
      });

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_LAMBDA(int iX) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(0, ilo - 1 - iX, iN) = uPF(0, ilo + iX, nNodes - iN - 1);
          uPF(0, ihi + 1 + iX, iN) = uPF(0, ihi - iX, nNodes - iN - 1);
        }
      });
}
