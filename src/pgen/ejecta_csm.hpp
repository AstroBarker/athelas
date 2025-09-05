#pragma once
/**
 * @file ejecta_csm.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Ejecta - CSM interaction test.
 * See Duffell 2016 (doi:10.3847/0004-637X/821/2/76)
 */

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * @brief Initialize ejecta csm test
 **/
void ejecta_csm_init(State* state, GridStructure* grid, ProblemIn* pin,
                     const EOS* eos, ModalBasis* fluid_basis = nullptr) {
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

  const auto rstar = pin->param()->get<double>("problem.params.rstar", 0.01);
  const auto vmax =
      pin->param()->get<double>("problem.params.vmax", std::sqrt(10.0 / 3.0));

  const double rstar3 = rstar * rstar * rstar;

  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;

  // Phase 1: Initialize nodal values (always done)
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          const double x = grid->node_coordinate(ix, iNodeX);
          if (x <= rstar) {
            uPF(ix, iNodeX, iPF_D) = 1.0 / (constants::FOURPI * rstar3 / 3.0);
          } else {
            uPF(ix, iNodeX, iPF_D) = 1.0;
          }
        }
      });

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
          const int k = 0;
          const double X1 = grid->get_centers(ix);

          if (X1 <= rstar) {
            const double rho = 1.0 / (constants::FOURPI * rstar3 / 3.0);
            const double pressure = (1.0e-5) * rho * vmax * vmax;
            const double vel = vmax * (X1 / rstar);
            uCF(ix, k, q_Tau) = 1.0 / rho;
            uCF(ix, k, q_V) = vel;
            uCF(ix, k, q_E) = (pressure / gm1 / rho) + 0.5 * vel * vel;
          } else {
            const double rho = 1.0;
            const double pressure = (1.0e-5) * rho * vmax * vmax;
            uCF(ix, k, q_Tau) = 1.0 / rho;
            uCF(ix, k, q_V) = 0.0;
            uCF(ix, k, q_E) = (pressure / gm1 / rho);
          }
        });
  }

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(ilo - 1 - ix, iN, 0) = uPF(ilo + ix, nNodes - iN - 1, 0);
          uPF(ihi + 1 + ix, iN, 0) = uPF(ihi - ix, nNodes - iN - 1, 0);
        }
      });
}
