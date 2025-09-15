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
void rad_advection_init(State *state, GridStructure *grid, ProblemIn *pin,
                        const EOS *eos, ModalBasis * /*fluid_basis = nullptr*/,
                        ModalBasis * /*radiation_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Radiation advection requires ideal gas eos!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();

  static const int ilo = 1;
  static const int ihi = grid->get_ihi();
  static const int nNodes = grid->get_n_nodes();

  const int q_Tau = 0;
  const int q_V = 1;
  const int q_E = 2;

  const int iPF_D = 0;

  const int iCR_E = 3;
  const int iCR_F = 4;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 1.0);
  const auto D = pin->param()->get<double>("problem.params.rho", 1.0);
  const auto amp = pin->param()->get<double>("problem.params.amp", 1.0);
  const auto width = pin->param()->get<double>("problem.params.width", 0.05);
  const double mu = 1.0 + constants::m_e / constants::m_p;
  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 2), KOKKOS_LAMBDA(int ix) {
        const int k = 0;
        const double X1 = grid->get_centers(ix);

        uCF(ix, k, iCR_E) =
            amp * std::max(std::exp(-std::pow((X1 - 0.5) / width, 2.0) / 2.0),
                           1.0e-8);
        uCF(ix, k, iCR_F) = 1.0 * constants::c_cgs * uCF(ix, k, iCR_E);

        const double Trad = std::pow(uCF(ix, k, iCR_E) / constants::a, 0.25);
        const double sie_fluid =
            constants::k_B * Trad / (gm1 * mu * constants::m_p);
        uCF(ix, k, q_Tau) = 1.0 / D;
        uCF(ix, k, q_V) = V0;
        uCF(ix, k, q_E) =
            sie_fluid +
            0.5 * V0 * V0; // p0 / (gamma - 1.0) / D + 0.5 * V0 * V0;

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
