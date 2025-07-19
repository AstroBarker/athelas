#pragma once
/**
 * @file boundary_conditions.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Boundary conditions
 *
 * @details Implemented BCs
 *            - outflow
 *            - reflecting
 *            - periodic
 *            - dirichlet
 */

#include "abstractions.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "grid.hpp"

namespace bc {
/**
 * @brief Apply Boundary Conditions to fluid fields
 *
 * Supported Options:
 *  outflow
 *  reflecting
 *  periodic
 *  dirichlet
 *  marshak
 **/
template <int N> // N = 3 for fluid, N = 2 for rad...
void fill_ghost_zones(View3D<double> U, const GridStructure* grid,
                      const ModalBasis* basis, BoundaryConditions* bcs) {

  const int nvars = U.extent(0);
  const int nX    = grid->get_n_elements();

  auto this_bc = get_bc_data<N>(bcs);

  Kokkos::parallel_for(
      "Fill ghost zones", nvars, KOKKOS_LAMBDA(const int q) {
        const int ghost_L    = 0;
        const int interior_L = (this_bc[0].type != BcType::Periodic) ? 1 : nX;
        const int ghost_R    = nX + 1;
        const int interior_R = (this_bc[1].type != BcType::Periodic) ? nX : 1;

        apply_bc<N>(this_bc[0], U, q, ghost_L, interior_L, basis);
        apply_bc<N>(this_bc[1], U, q, ghost_R, interior_R, basis);
      });
}

// Applies boundary condition for one variable `q`
template <int N>
KOKKOS_INLINE_FUNCTION void
apply_bc(const BoundaryConditionsData<N>& bc, View3D<double> U, const int q,
         const int ghost_cell, const int interior_cell,
         const ModalBasis* basis) {
  const int num_modes = basis->get_order();
  switch (bc.type) {
  case BcType::Outflow:
    for (int k = 0; k < num_modes; k++) {
      U(q, ghost_cell, k) = U(q, interior_cell, k);
    }
    break;

  // NOTE: Literally the same as the above, but the required
  // use is different. interior_cell should be of the opposite
  // side as ghost_cell. Not ideal, but works.
  case BcType::Periodic:
    // assert( interior_cell != ghost_cell + 1 && "Bad use of periodic BC!\n" );
    // assert( interior_cell != ghost_cell - 1 && "Bad use of periodic BC!\n" );
    for (int k = 0; k < num_modes; k++) {
      U(q, ghost_cell, k) = U(q, interior_cell, k);
    }
    break;

  case BcType::Reflecting:
    for (int k = 0; k < num_modes; k++) {
      if (q == 1) { // Momentum (q == 1)
        // Reflect momentum in the cell average (k == 0) and leave higher modes
        // unchanged (k > 0)
        U(q, ghost_cell, k) =
            (k == 0) ? -U(q, interior_cell, k) : U(q, interior_cell, k);
      } else { // Non-momentum variables
        // Reflect cell averages (k == 0) and invert higher modes (k > 0)
        U(q, ghost_cell, k) =
            (k == 0) ? U(q, interior_cell, k) : -U(q, interior_cell, k);
      }
    }
    break;

  // TODO(astrobarker): could need extending. FIX
  case BcType::Dirichlet:
    // U(q, ghost_cell, 0) = 2.0 * bc.dirichlet_values[q] - U(q, interior_cell,
    // 0);
    U(q, ghost_cell, 0) = bc.dirichlet_values[q];
    for (int k = 1; k < num_modes; k++) {
      U(q, ghost_cell, k) = 0.0; // slopes++ set to 0
    }
    break;

  // FIX: check if slope++ are correct.
  case BcType::Marshak: {
    // Marshak uses dirichlet_values
    const double Einc = bc.dirichlet_values[0]; // aT^4
    for (int k = 0; k < 1; k++) {
      if (q == 0) {
        if (k == 0) {
          U(q, ghost_cell, k) = (k == 0) ? Einc : 0;
        }
      } else if (q == 1) {
        constexpr static double c = constants::c_cgs;
        const double E0           = U(0, interior_cell, k);
        const double F0           = U(1, interior_cell, k);
        U(q, ghost_cell, k) =
            (k == 0) ? 0.5 * c * Einc - 0.5 * (c * E0 + 2.0 * F0) : 0.0;
      }
    }
  } // case Marshak
  break;

  // formality
  case BcType::Null:
    // TODO(astrobarker) not okay for device
    THROW_ATHELAS_ERROR("Null BC is not for use!");
    break;
  }
}
} // namespace bc
