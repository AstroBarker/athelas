#pragma once
/**
 * @file rad_equilibrium.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation fluid equilibriation test
 */

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * Initialize equilibrium rad test
 **/
void rad_equilibrium_init(State* state, GridStructure* grid, ProblemIn* pin,
                          const EOS* /*eos*/,
                          ModalBasis* /*fluid_basis = nullptr*/,
                          ModalBasis* /*radiation_basis = nullptr*/) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  if (!rad_active) {
    THROW_ATHELAS_ERROR("Radiation equilibriation requires radiation enabled!");
  }
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Radiation equilibriation requires ideal gas eos!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();

  const int ilo = 1;
  const int ihi = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  const int iCF_Tau = 0;
  const int iCF_V = 1;
  const int iCF_E = 2;

  const int iPF_D = 0;

  const int iCR_E = 3;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto logD = pin->param()->get<double>("problem.params.logrho", -7.0);
  const auto logE_gas =
      pin->param()->get<double>("problem.params.logE_gas", 10.0);
  const auto logE_rad =
      pin->param()->get<double>("problem.params.logE_rad", 12.0);

  const double D = std::pow(10.0, logD);
  const double Ev_gas = std::pow(10.0, logE_gas);
  const double Ev_rad = std::pow(10.0, logE_rad);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 2), KOKKOS_LAMBDA(int iX) {
        const int k = 0;

        uCF(iCF_Tau, iX, k) = 1.0 / D;
        uCF(iCF_V, iX, k) = V0;
        uCF(iCF_E, iX, k) = Ev_gas / D;
        uCF(iCR_E, iX, k) = Ev_rad;

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
