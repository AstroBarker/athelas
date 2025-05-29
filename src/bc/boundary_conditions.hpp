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
 **/
template <int N> // N = 3 for fluid, N = 2 for rad...
void fill_ghost_zones( View3D<Real> U, const GridStructure* grid,
                       const int order, BoundaryConditions* bcs ) {

  const int nvars = U.extent( 0 );
  const int nX    = grid->get_n_elements( );

  auto this_bc = get_bc_data<N>( bcs );

  Kokkos::parallel_for(
      "Fill ghost zones", nvars, KOKKOS_LAMBDA( const int q ) {
        const int ghost_L    = 0;
        const int interior_L = (this_bc[0].type != BcType::Periodic) ? 1 :nX;
        const int ghost_R    = nX + 1;
        const int interior_R = (this_bc[1].type != BcType::Periodic) ? nX : 1;

        apply_bc<N>( this_bc[0], U, q, ghost_L, interior_L, order );
        apply_bc<N>( this_bc[1], U, q, ghost_R, interior_R, order );
      } );
}
} // namespace bc
