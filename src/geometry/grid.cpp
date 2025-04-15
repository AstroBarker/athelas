/**
 * @file grid.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Class for holding the spatial grid.
 *
 * @details This class GridStructure holds key pieces of the grid:
 *          - nx
 *          - nnodes
 *          - weights
 *
 *          For a loop over real zones, loop from ilo to ihi (inclusive).
 *          ilo = nGhost_
 *          ihi = nElements_ - nGhost_ + 1
 */

#include <cmath> /* atan */
#include <vector>

#include "constants.hpp"
#include "grid.hpp"

GridStructure::GridStructure( const ProblemIn* pin )
    : nElements_( pin->nElements ), nNodes_( pin->nNodes ),
      nGhost_( pin->nGhost ), mSize_( nElements_ + ( 2 * nGhost_ ) ),
      xL_( pin->xL ), xR_( pin->xR ), geometry_( pin->Geometry ),
      nodes_( "Nodes", pin->nNodes ), weights_( "weights_", pin->nNodes ),
      centers_( "Cetners", mSize_ ), widths_( "widths_", mSize_ ),
      x_l_( "Left Interface", mSize_ + 1 ), mass_( "Cell mass_", mSize_ ),
      center_of_mass_( "Center of mass_", mSize_ ),
      grid_( "Grid", mSize_, nNodes_ ) {
  std::vector<Real> tmp_nodes( nNodes_ );
  std::vector<Real> tmp_weights( nNodes_ );

  for ( int iN = 0; iN < nNodes_; iN++ ) {
    tmp_nodes[iN]   = 0.0;
    tmp_weights[iN] = 0.0;
  }

  quadrature::lg_quadrature( nNodes_, tmp_nodes, tmp_weights );

  for ( int iN = 0; iN < nNodes_; iN++ ) {
    nodes_( iN )   = tmp_nodes[iN];
    weights_( iN ) = tmp_weights[iN];
  }

  create_grid( );
}

// linear shape function on the reference element
KOKKOS_INLINE_FUNCTION
const Real shape_function( const int interface, const Real eta ) {
  Real mult = 1.0;

  if ( interface == 0 ) {
    mult = -1.0;
  }
  if ( interface == 1 ) {
    mult = +1.0;
  }
  if ( interface != 0 && interface != 1 ) {
    THROW_ATHELAS_ERROR( " ! Invalid shape func params" );
  }

  return 0.5 + ( mult * eta );
}

// Give physical grid coordinate from a node.
auto GridStructure::node_coordinate( int iC, int iN ) const -> Real {
  return x_l_( iC ) * shape_function( 0, nodes_( iN ) ) +
         x_l_( iC + 1 ) * shape_function( 1, nodes_( iN ) );
}

// Return cell center
auto GridStructure::get_centers( int iC ) const -> Real {
  return centers_( iC );
}

// Return cell width
auto GridStructure::get_widths( int iC ) const -> Real { return widths_( iC ); }

// Return cell mass
auto GridStructure::get_mass( int iX ) const -> Real { return mass_( iX ); }

// Return cell reference Center of mass_
auto GridStructure::get_center_of_mass( int iX ) const -> Real {
  return center_of_mass_( iX );
}

// Return given quadrature node
auto GridStructure::get_nodes( int nN ) const -> Real { return nodes_( nN ); }

// Return given quadrature weight
auto GridStructure::get_weights( int nN ) const -> Real {
  return weights_( nN );
}

// Acessor for xL
auto GridStructure::get_x_l( ) const noexcept -> Real { return xL_; }

// Acessor for xR
auto GridStructure::get_x_r( ) const noexcept -> Real { return xR_; }

// Acessor for SqrtGm
auto GridStructure::get_sqrt_gm( Real X ) const -> Real {
  if ( geometry_ == geometry::Spherical ) {
    return X * X;
  }
  return 1.0;
}

// Accessor for x_l_
auto GridStructure::get_left_interface( int iX ) const -> Real {
  return x_l_( iX );
}

// Return nNodes_
auto GridStructure::get_n_nodes( ) const noexcept -> int { return nNodes_; }

// Return nElements_
auto GridStructure::get_n_elements( ) const noexcept -> int {
  return nElements_;
}

// Return number of guard zones
auto GridStructure::get_guard( ) const noexcept -> int { return nGhost_; }

// Return first physical zone
auto GridStructure::get_ilo( ) const noexcept -> int { return nGhost_; }

// Return last physical zone
auto GridStructure::get_ihi( ) const noexcept -> int {
  return nElements_ + nGhost_ - 1;
}

// Return true if in spherical symmetry
auto GridStructure::do_geometry( ) const noexcept -> bool {
  return geometry_ == geometry::Spherical;
}

// Equidistant mesh
// TODO(astrobarker): We will need to replace centers_ here, right?
void GridStructure::create_grid( ) {

  const int ilo = nGhost_; // first real zone
  const int ihi = nElements_ + nGhost_ - 1; // last real zone

  for ( int i = 0; i < nElements_ + 2 * nGhost_; i++ ) {
    widths_( i ) = ( xR_ - xL_ ) / nElements_;
  }

  x_l_( nGhost_ ) = xL_;
  for ( int iX = 2; iX < nElements_ + 2 * nGhost_; iX++ ) {
    x_l_( iX ) = x_l_( iX - 1 ) + widths_( iX - 1 );
  }

  centers_( ilo ) = xL_ + 0.5 * widths_( ilo );
  for ( int i = ilo + 1; i <= ihi; i++ ) {
    centers_( i ) = centers_( i - 1 ) + widths_( i - 1 );
  }

  for ( int i = ilo - 1; i >= 0; i-- ) {
    centers_( i ) = centers_( i + 1 ) - widths_( i + 1 );
  }
  for ( int i = ihi + 1; i < nElements_ + nGhost_ + 1; i++ ) {
    centers_( i ) = centers_( i - 1 ) + widths_( i - 1 );
  }

  for ( int iC = ilo; iC <= ihi; iC++ ) {
    for ( int iN = 0; iN < nNodes_; iN++ ) {
      grid_( iC, iN ) = node_coordinate( iC, iN );
    }
  }
}

/**
 * Compute cell masses
 **/
void GridStructure::compute_mass( View3D<Real> uPF ) {
  const int nNodes_ = get_n_nodes( );
  const int ilo     = get_ilo( );
  const int ihi     = get_ihi( );

  Real mass = NAN;
  Real X    = NAN;

  for ( int iX = ilo; iX <= ihi; iX++ ) {
    mass = 0.0;
    for ( int iN = 0; iN < nNodes_; iN++ ) {
      X = node_coordinate( iX, iN );
      mass += weights_( iN ) * get_sqrt_gm( X ) * uPF( 0, iX, iN );
    }
    mass *= widths_( iX );
    mass_( iX ) = mass;
  }

  // Guard cells
  for ( int iX = 0; iX < ilo; iX++ ) {
    mass_( ilo - 1 - iX ) = mass_( ilo + iX );
    mass_( ihi + 1 + iX ) = mass_( ihi - iX );
  }
}

/**
 * Compute cell centers of masses reference coordinates
 **/
void GridStructure::compute_center_of_mass( View3D<Real> uPF ) {
  const int nNodes_ = get_n_nodes( );
  const int ilo     = get_ilo( );
  const int ihi     = get_ihi( );

  Real com = 0.0;
  Real X   = 0.0;

  for ( int iX = ilo; iX <= ihi; iX++ ) {
    com = 0.0;
    for ( int iN = 0; iN < nNodes_; iN++ ) {
      X = node_coordinate( iX, iN );
      com +=
          nodes_( iN ) * weights_( iN ) * get_sqrt_gm( X ) * uPF( 0, iX, iN );
    }
    com *= widths_( iX );
    center_of_mass_( iX ) = com / mass_( iX );
  }

  // Guard cells
  for ( int iX = 0; iX < ilo; iX++ ) {
    center_of_mass_( ilo - 1 - iX ) = center_of_mass_( ilo + iX );
    center_of_mass_( ihi + 1 + iX ) = center_of_mass_( ihi - iX );
  }
}

/**
 * Update grid coordinates using interface velocities.
 **/
void GridStructure::update_grid( View1D<Real> SData ) {

  const int ilo = get_ilo( );
  const int ihi = get_ihi( );

  Kokkos::parallel_for(
      "Grid Update 1", Kokkos::RangePolicy<>( ilo, ihi + 2 ),
      KOKKOS_CLASS_LAMBDA( int iX ) {
        x_l_( iX )     = SData( iX );
        widths_( iX )  = SData( iX + 1 ) - SData( iX );
        centers_( iX ) = 0.5 * ( SData( iX + 1 ) + SData( iX ) );
      } );

  Kokkos::parallel_for(
      "Grid Update 2", Kokkos::RangePolicy<>( ilo, ihi + 2 ),
      KOKKOS_CLASS_LAMBDA( int iX ) {
        for ( int iN = 0; iN < nNodes_; iN++ ) {
          grid_( iX, iN ) = node_coordinate( iX, iN );
        }
      } );
}

// Access by (element, node)
auto GridStructure::operator( )( int i, int j ) -> Real& {
  return grid_( i, j );
}

auto GridStructure::operator( )( int i, int j ) const -> Real {
  return grid_( i, j );
}
