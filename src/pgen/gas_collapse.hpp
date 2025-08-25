#pragma once
/**
 * @file gas_collapse.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Collapsing gas cloud
 */

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * @brief Initialize gas collapse
 **/
void gas_collapse_init(State* state, GridStructure* grid, ProblemIn* pin,
                       const EOS* eos, ModalBasis* /*fluid_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Gas collapse requires ideal gas eos!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();

  const int ilo = 1;
  const int ihi = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V = 1;
  constexpr static int iCF_E = 2;

  constexpr static int iPF_D = 0;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto p0 = pin->param()->get<double>("problem.params.p0", 10.0);

  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int iX) {
        const int k = 0;

        uCF(iX, k, iCF_Tau) = rho0; // / rho0 * (1.0 / std::cosh(x / H));
        uCF(iX, k, iCF_V) = V0;
        uCF(iX, k, iCF_E) = (p0 / gm1) * uCF(iX, k, iCF_Tau) + 0.5 * V0 * V0;

        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(iX, iNodeX, iPF_D) = rho0;
        }
      });

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_LAMBDA(int iX) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(ilo - 1 - iX, iN, 0) = uPF(ilo + iX, nNodes - iN - 1, 0);
          uPF(ilo + 1 + iX, iN, 0) = uPF(ilo - iX, nNodes - iN - 1, 0);
        }
      });
}
