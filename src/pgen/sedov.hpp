#pragma once
/**
 * @file sedov.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Sedov blast wave
 */

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * @brief Initialize sedov blast wave
 **/
void sedov_init(State* state, GridStructure* grid, ProblemIn* pin,
                const EOS* eos, ModalBasis* /*fluid_basis = nullptr*/) {

  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Sedov requires ideal gas eos!");
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
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int iX) {
        const int k = 0;

        uCF(iCF_Tau, iX, k) = 1.0 / D0;
        uCF(iCF_V, iX, k) = V0;
        if (iX == origin - 1 || iX == origin) {
          uCF(iCF_E, iX, k) = (P0 / gm1) * uCF(iCF_Tau, iX, k) + 0.5 * V0 * V0;
        } else {
          uCF(iCF_E, iX, k) =
              (1.0e-6 / gm1) * uCF(iCF_Tau, iX, k) + 0.5 * V0 * V0;
        }

        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(iPF_D, iX, iNodeX) = D0;
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
