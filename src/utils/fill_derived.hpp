#pragma once

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * @brief fill derived quantities
 *
 * Populates primitive and auxilliary fields.
 * TODO(astrobarker): expand once we want to add more quantities.
 */
inline void fill_derived(const State* state, const EOS* eos,
                         const GridStructure* grid,
                         const ModalBasis* fluid_basis) {
  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();
  View3D<double> uAF = state->u_af();

  const int ilo = 1;
  const int ihi = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes; ++iN) {
          const double tau = fluid_basis->basis_eval(uCF, ix, 0, iN + 1);
          const double vel = fluid_basis->basis_eval(uCF, ix, 1, iN + 1);
          const double emt = fluid_basis->basis_eval(uCF, ix, 2, iN + 1);

          const double rho = 1.0 / tau;
          const double momentum = rho * vel;
          const double sie = (emt - 0.5 * vel * vel);

          auto lambda = nullptr;
          const double pressure =
              pressure_from_conserved(eos, tau, vel, emt, lambda);
          const double t_gas =
              temperature_from_conserved(eos, tau, vel, emt, lambda);

          uPF(ix, iN, 0) = rho;
          uPF(ix, iN, 1) = momentum;
          uPF(ix, iN, 2) = sie;

          uAF(ix, iN, 0) = pressure;
          uAF(ix, iN, 1) = t_gas;
        }
      });
}
