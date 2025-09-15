#pragma once
/**
 * @file rad_wave.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation wave test
 */

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * @brief Initialize radiation wave test
 **/
void rad_wave_init(State *state, GridStructure *grid, ProblemIn *pin,
                   const EOS *eos, ModalBasis * /*fluid_basis = nullptr*/,
                   ModalBasis * /*radiation_basis = nullptr*/) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  if (!rad_active) {
    THROW_ATHELAS_ERROR("Radiation wave requires radiation enabled!");
  }

  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Radiation wave requires ideal gas eos!");
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

  constexpr static int iCR_E = 3;

  const auto lambda = pin->param()->get<double>("problem.params.lambda", 0.1);
  const auto kappa = pin->param()->get<double>("problem.params.kappa", 1.0);
  const auto epsilon =
      pin->param()->get<double>("problem.params.epsilon", 1.0e-6);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto P0 = pin->param()->get<double>("problem.params.p0", 1.0e-6);

  // TODO(astrobarker): thread through
  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 2), KOKKOS_LAMBDA(int ix) {
        const int k = 0;
        const double X1 = grid->get_centers(ix);

        uCF(ix, k, q_Tau) = 1.0 / rho0;
        uCF(ix, k, q_V) = 0.0;
        uCF(ix, k, q_E) = (P0 / gm1) / rho0;
        uCF(ix, k, iCR_E) = epsilon;

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
