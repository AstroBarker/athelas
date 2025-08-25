#pragma once
/**
 * @file smooth_flow.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Smooth flow test
 */

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"
#include "utils/constants.hpp"

/**
 * @brief Initialize smooth flow test problem
 **/
void smooth_flow_init(State* state, GridStructure* grid, ProblemIn* pin,
                      const EOS* /*eos*/, ModalBasis* fluid_basis = nullptr) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Smooth flow requires ideal gas eos!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();

  const int ilo = 1;
  const int ihi = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto amp =
      pin->param()->get<double>("problem.params.amp", 0.9999999999999999);

  // Phase 1: Initialize nodal values (always done)
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          const double x = grid->node_coordinate(ix, iNodeX);
          uPF(ix, iNodeX, iPF_D) = (1.0 + amp * sin(constants::PI * x));
        }
      });

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    // Use L2 projection for accurate modal coefficients
    auto density_func = [&amp](double x, int /*ix*/, int /*iN*/) -> double {
      return 1.0 + amp * sin(constants::PI * x);
    };

    auto velocity_func = [](double /*x*/, int /*ix*/, int /*iN*/) -> double {
      return 0.0;
    };

    auto energy_func = [&amp](double x, int /*ix*/, int /*iN*/) -> double {
      const double D = 1.0 + amp * sin(constants::PI * x);
      return (D * D * D / 2.0) / D;
    };

    // Project each conserved variable using Kokkos parallel for
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

          const double D = (1.0 + (amp * sin(constants::PI * X1)));
          uCF(ix, k, q_Tau) = 1.0 / D;
          uCF(ix, k, q_V) = 0.0;
          uCF(ix, k, q_E) = (D * D * D / 2.0) * uCF(ix, k, q_Tau);
        });
  }

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(ilo - 1 - ix, iN, 0) = uPF(ilo + ix, nNodes - iN - 1, 0);
          uPF(ilo + 1 + ix, iN, 0) = uPF(ilo - ix, nNodes - iN - 1, 0);
        }
      });
}
