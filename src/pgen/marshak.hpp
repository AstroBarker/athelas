#pragma once
/**
 * @file marshak.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation marshak wave test
 */

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"
#include "utils/constants.hpp"

/**
 * @brief Initialize radiating shock
 **/
void marshak_init(State* state, GridStructure* grid, ProblemIn* pin,
                  const EOS* /*eos*/, ModalBasis* /*fluid_basis = nullptr*/,
                  ModalBasis* /*radiation_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "marshak") {
    THROW_ATHELAS_ERROR("Marshak requires marshak eos!");
  }

  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  if (!rad_active) {
    THROW_ATHELAS_ERROR("Marshak requires radiation enabled!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();

  const int ilo = 1;
  const int ihi = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  // TODO(astrobarker) move these to a namespace like constants
  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V = 1;
  constexpr static int iCF_E = 2;

  constexpr static int iPF_D = 0;

  constexpr static int iCR_E = 3;
  constexpr static int iCR_F = 4;

  auto su_olson_energy = [&](const double alpha, const double T) {
    return (alpha / 4.0) * std::pow(T, 4.0);
  };

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 10.0);
  const auto epsilon = pin->param()->get<double>("problem.params.epsilon", 1.0);
  const auto T0 = pin->param()->get<double>("problem.params.T0", 1.0e4); // K

  const double alpha = 4.0 * constants::a / epsilon;
  const double em_gas = su_olson_energy(alpha, T0) / rho0;

  // TODO(astrobarker): thread through
  const double e_rad = constants::a * std::pow(T0, 4.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 2), KOKKOS_LAMBDA(int iX) {
        const int k = 0;

        uCF(iCF_Tau, iX, k) = 1.0 / rho0;
        uCF(iCF_V, iX, k) = V0;
        uCF(iCF_E, iX, k) = em_gas + 0.5 * V0 * V0;
        uCF(iCR_E, iX, k) = e_rad;
        uCF(iCR_F, iX, k) = 0.0;

        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(iPF_D, iX, iNodeX) = rho0;
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
