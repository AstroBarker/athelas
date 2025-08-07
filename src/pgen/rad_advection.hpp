#pragma once
/**
 * @file rad_advection.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation advection test
 */

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"
#include "utils/constants.hpp"

/**
 * @brief Initialize radiation advection test
 * @note EXPERIMENTAL
 **/
void rad_advection_init(State* state, GridStructure* grid, ProblemIn* pin,
                        const EOS* eos, ModalBasis* /*fluid_basis = nullptr*/,
                        ModalBasis* /*radiation_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Radiation advection requires ideal gas eos!");
  }

  View3D<double> uCF = state->get_u_cf();
  View3D<double> uPF = state->get_u_pf();

  const int ilo    = 1;
  const int ihi    = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  const int iCF_Tau = 0;
  const int iCF_V   = 1;
  const int iCF_E   = 2;

  const int iPF_D = 0;

  const int iCR_E = 3;
  const int iCR_F = 4;

  const auto V0      = pin->param()->get<double>("problem.params.v0", 1.0);
  const auto D       = pin->param()->get<double>("problem.params.rho", 1.0);
  const auto amp     = pin->param()->get<double>("problem.params.amp", 1.0);
  const auto width   = pin->param()->get<double>("problem.params.width", 0.05);
  const double mu    = 1.0 + constants::m_e / constants::m_p;
  const double gamma = get_gamma(eos);
  const double gm1   = gamma - 1.0;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 2), KOKKOS_LAMBDA(int iX) {
        const int k     = 0;
        const double X1 = grid->get_centers(iX);

        uCF(iCR_E, iX, k) =
            amp * std::max(std::exp(-std::pow((X1 - 0.5) / width, 2.0) / 2.0),
                           1.0e-8);
        uCF(iCR_F, iX, k) = 1.0 * constants::c_cgs * uCF(iCR_E, iX, k);

        const double Trad = std::pow(uCF(iCR_E, iX, k) / constants::a, 0.25);
        const double sie_fluid =
            constants::k_B * Trad / (gm1 * mu * constants::m_p);
        uCF(iCF_Tau, iX, k) = 1.0 / D;
        uCF(iCF_V, iX, k)   = V0;
        uCF(iCF_E, iX, k) =
            sie_fluid +
            0.5 * V0 * V0; // p0 / (gamma - 1.0) / D + 0.5 * V0 * V0;

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
