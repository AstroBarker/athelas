#pragma once
/**
 * @file advection.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Fluid advection test
 */

#include <cmath> /* sin */

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"
#include "utils/constants.hpp"

/**
 * Initialize advection test
 **/
void advection_init(State* state, GridStructure* grid, ProblemIn* pin,
                    const EOS* eos, ModalBasis* fluid_basis = nullptr) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Advection requires ideal gas eos!");
  }

  // Smooth advection problem
  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();

  static const int ilo = 1;
  static const int ihi = grid->get_ihi();
  static const int nNodes = grid->get_n_nodes();

  const int q_Tau = 0;
  const int q_V = 1;
  const int q_E = 2;

  const int iPF_D = 0;

  const auto V0 = pin->param()->get<double>("problem.params.v0", -1.0);
  const auto P0 = pin->param()->get<double>("problem.params.p0", 0.01);
  const auto Amp = pin->param()->get<double>("problem.params.amp", 1.0);

  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;

  // Phase 1: Initialize nodal values (always done)
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          const double x = grid->node_coordinate(ix, iNodeX);
          uPF(ix, iNodeX, iPF_D) = (2.0 + Amp * sin(2.0 * constants::PI * x));
        }
      });

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    // Use L2 projection for accurate modal coefficients
    auto density_func = [&Amp](double x, int /*ix*/, int /*iN*/) -> double {
      return 2.0 + Amp * sin(2.0 * constants::PI * x);
    };

    auto velocity_func = [&V0](double /*x*/, int /*ix*/, int /*iN*/) -> double {
      return V0;
    };

    auto energy_func = [&P0, &V0, &Amp, &gm1](double x, int /*ix*/,
                                              int /*iN*/) -> double {
      const double rho = 2.0 + Amp * sin(2.0 * constants::PI * x);
      return (P0 / gm1) / rho + 0.5 * V0 * V0;
    };

    // L2 projection onto modal basis
    fluid_basis->project_nodal_to_modal_all_cells(uCF, uPF, grid, q_Tau,
                                                  density_func);
    fluid_basis->project_nodal_to_modal_all_cells(uCF, uPF, grid, q_V,
                                                  velocity_func);
    fluid_basis->project_nodal_to_modal_all_cells(uCF, uPF, grid, q_E,
                                                  energy_func);
  } else {
    // Fallback: set cell averages only (k=0)
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
          const int k = 0;
          const double X1 = grid->get_centers(ix);

          uCF(ix, k, q_Tau) = 1.0 / (2.0 + Amp * sin(2.0 * constants::PI * X1));
          uCF(ix, k, q_V) = V0;
          uCF(ix, k, q_E) = (P0 / gm1) * uCF(ix, k, q_Tau) + 0.5 * V0 * V0;
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
