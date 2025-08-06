#pragma once
/**
 * @file rad_wave.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation wave test
 */

#include "abstractions.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize radiation wave test
 **/
void rad_wave_init(State* state, GridStructure* grid, ProblemIn* pin) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  if (!rad_active) {
    THROW_ATHELAS_ERROR("Radiation wave requires radiation enabled!");
  }

  View3D<double> uCF = state->get_u_cf();
  View3D<double> uPF = state->get_u_pf();

  const int ilo    = grid->get_ilo();
  const int ihi    = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  constexpr static int iCR_E = 3;

  const auto lambda = pin->param()->get<double>("problem.params.lambda", 0.1);
  const auto kappa  = pin->param()->get<double>("problem.params.kappa", 1.0);
  const auto epsilon =
      pin->param()->get<double>("problem.params.epsilon", 1.0e-6);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto P0   = pin->param()->get<double>("problem.params.p0", 1.0e-6);

  // TODO(astrobarker): thread through
  const double gamma = 5.0 / 3.0;
  const double gm1   = gamma - 1.0;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 2),
      KOKKOS_LAMBDA(int iX) {
        const int k = 0;
        const double X1 = grid->get_centers(iX);

        uCF(iCF_Tau, iX, k) = 1.0 / rho0;
        uCF(iCF_V, iX, k)   = 0.0;
        uCF(iCF_E, iX, k)   = (P0 / gm1) / rho0;

        uCF(iCR_E, iX, k) = epsilon;

        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(iPF_D, iX, iNodeX) = rho0;
        }
      });

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo),
      KOKKOS_LAMBDA(int iX) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(0, ilo - 1 - iX, iN) = uPF(0, ilo + iX, nNodes - iN - 1);
          uPF(0, ihi + 1 + iX, iN) = uPF(0, ihi - iX, nNodes - iN - 1);
        }
      });
}
