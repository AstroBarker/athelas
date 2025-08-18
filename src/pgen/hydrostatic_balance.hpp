#pragma once
/**
 * @file hydrostatic_balance.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Shu Osher shock tube
 */

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "solvers/hydrostatic_equilibrium.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * @brief Initialize Shu Osher hydro test
 **/
void hydrostatic_balance_init(State* state, GridStructure* grid, ProblemIn* pin,
                              const EOS* eos,
                              ModalBasis* fluid_basis = nullptr) {
  if (pin->param()->get<std::string>("eos.type") != "polytropic") {
    THROW_ATHELAS_ERROR("Hydrostatic balance requires polytropic eos!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();
  View3D<double> uAF = state->u_af();

  const int ilo    = 1;
  const int ihi    = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  const auto rho_c = pin->param()->get<double>("problem.params.rho_c", 1.0e8);
  const auto p_thresh =
      pin->param()->get<double>("problem.params.p_threshold", 1.0e-10);

  const auto polytropic_k = pin->param()->get<double>("eos.k");
  const auto polytropic_n = pin->param()->get<double>("eos.n");

  const double gamma = get_gamma(eos);
  const double gm1   = gamma - 1.0;

  auto rho_from_p = [&polytropic_k, &polytropic_n](const double p) -> double {
    return std::pow(p / polytropic_k, polytropic_n / (polytropic_n + 1.0));
  };

  if (fluid_basis == nullptr) {
    auto solver = HydrostaticEquilibrium(rho_c, p_thresh, eos,
                                         pin->param()->get<double>("eos.k"),
                                         pin->param()->get<double>("eos.n"));
    solver.solve(state->u_af(), grid, pin);

    // Phase 1: Initialize nodal values (always done)
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int iX) {
          for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
            uPF(iPF_D, iX, iNodeX) = rho_from_p(uAF(0, iX, iNodeX));
          }
        });
  }

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    // Use L2 projection for accurate modal coefficients
    auto tau_func = [&](double /*x*/, int iX, int iN) -> double {
      return 1.0 / rho_from_p(uAF(0, iX, iN));
    };

    auto velocity_func = [](double /*x*/, int /*iX*/, int /*iN*/) -> double {
      return 0.0;
    };

    auto energy_func = [&](double /*x*/, int iX, int iN) -> double {
      const double rho = rho_from_p(uAF(0, iX, iN));
      return (uAF(0, iX, iN) / gm1) / rho;
    };

    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int iX) {
          const int k     = 0;
          const double X1 = grid->get_centers(iX);

          // Project each conserved variable
          fluid_basis->project_nodal_to_modal(uCF, uPF, grid, iCF_Tau, iX,
                                              tau_func);
          fluid_basis->project_nodal_to_modal(uCF, uPF, grid, iCF_V, iX,
                                              velocity_func);
          fluid_basis->project_nodal_to_modal(uCF, uPF, grid, iCF_E, iX,
                                              energy_func);
        });
  }

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_LAMBDA(int iX) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(0, ilo - 1 - iX, iN) = uPF(0, ilo + iX, nNodes - iN - 1);
          uPF(0, ihi + 1 + iX, iN) = uPF(0, ihi - iX, nNodes - iN - 1);
        }
      });
}
