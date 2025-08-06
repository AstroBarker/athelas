#pragma once
/**
 * @file rad_shock.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation shock test
 */

#include <cmath> /* sin */

#include "abstractions.hpp"
#include "constants.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize radiating shock
 **/
void rad_shock_init(State* state, GridStructure* grid, ProblemIn* pin,
                    ModalBasis* fluid_basis = nullptr, ModalBasis* radiation_basis = nullptr) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  if (!rad_active) {
    THROW_ATHELAS_ERROR("Radiative shock requires radiation enabled!");
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

  const auto V_L  = pin->param()->get<double>("problem.params.vL", 5.19e7);
  const auto V_R  = pin->param()->get<double>("problem.params.vR", 1.73e7);
  const auto rhoL = pin->param()->get<double>("problem.params.rhoL", 5.69);
  const auto rhoR = pin->param()->get<double>("problem.params.rhoR", 17.1);
  const auto T_L = pin->param()->get<double>("problem.params.T_L", 2.18e6); // K
  const auto T_R = pin->param()->get<double>("problem.params.T_R", 7.98e6); // K
  const auto x_d = pin->param()->get<double>("problem.params.x_d", 0.013);

  // TODO(astrobarker): thread through
  const double mu       = 1.0 + constants::m_e / constants::m_p;
  const double gamma    = 5.0 / 3.0;
  const double gm1      = gamma - 1.0;
  const double em_gas_L = constants::k_B * T_L / (gm1 * mu * constants::m_p);
  const double em_gas_R = constants::k_B * T_R / (gm1 * mu * constants::m_p);
  const double e_rad_L  = constants::a * std::pow(T_L, 4.0);
  const double e_rad_R  = constants::a * std::pow(T_R, 4.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 2),
      KOKKOS_LAMBDA(int iX) {
        const int k = 0;
        const double X1 = grid->get_centers(iX);

        if (X1 <= x_d) {
          uCF(iCF_Tau, iX, k) = 1.0 / rhoL;
          uCF(iCF_V, iX, k)   = V_L;
          uCF(iCF_E, iX, k)   = em_gas_L + 0.5 * V_L * V_L;
          uCF(iCR_E, iX, k)   = e_rad_L;

          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(iPF_D, iX, iNodeX) = rhoL;
          }
        } else {
          uCF(iCF_Tau, iX, k) = 1.0 / rhoR;
          uCF(iCF_V, iX, k)   = V_R;
          uCF(iCF_E, iX, k)   = em_gas_R + 0.5 * V_R * V_R;
          uCF(iCR_E, iX, k)   = e_rad_R;

          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(iPF_D, iX, iNodeX) = rhoR;
          }
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
