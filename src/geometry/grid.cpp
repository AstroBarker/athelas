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
 *          ilo = nGhost
 *          ihi = nElements - nGhost + 1
 */

#include <cmath> /* atan */

#include "constants.hpp"
#include "grid.hpp"

GridStructure::GridStructure( const ProblemIn* pin )
    : nElements( pin->nElements ), nNodes( pin->nNodes ), nGhost( pin->nGhost ),
      mSize( nElements + ( 2 * nGhost ) ), xL( pin->xL ), xR( pin->xR ),
      Geometry( pin->Geometry ), Nodes( "Nodes", pin->nNodes ),
      Weights( "Weights", pin->nNodes ), Centers( "Cetners", mSize ),
      Widths( "Widths", mSize ), X_L( "Left Interface", mSize + 1 ),
      Mass( "Cell Mass", mSize ), CenterOfMass( "Center of Mass", mSize ),
      Grid( "Grid", mSize, nNodes ) {
  // TODO(astrobarker): Allow lg_quadrature to take in vectors.
  Real* tmp_nodes   = new Real[nNodes];
  Real* tmp_weights = new Real[nNodes];

  for ( int iN = 0; iN < nNodes; iN++ ) {
    tmp_nodes[iN]   = 0.0;
    tmp_weights[iN] = 0.0;
  }

  quadrature::lg_quadrature( nNodes, tmp_nodes, tmp_weights );

  for ( int iN = 0; iN < nNodes; iN++ ) {
    Nodes( iN )   = tmp_nodes[iN];
    Weights( iN ) = tmp_weights[iN];
  }

  create_grid( );

  delete[] tmp_nodes;
  delete[] tmp_weights;
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
  return X_L( iC ) * shape_function( 0, Nodes( iN ) ) +
         X_L( iC + 1 ) * shape_function( 1, Nodes( iN ) );
}

// Return cell center
auto GridStructure::get_centers( int iC ) const -> Real {
  return Centers( iC );
}

// Return cell width
auto GridStructure::get_widths( int iC ) const -> Real { return Widths( iC ); }

// Return cell mass
auto GridStructure::get_mass( int iX ) const -> Real { return Mass( iX ); }

// Return cell reference Center of Mass
auto GridStructure::get_center_of_mass( int iX ) const -> Real {
  return CenterOfMass( iX );
}

// Return given quadrature node
auto GridStructure::get_nodes( int nN ) const -> Real { return Nodes( nN ); }

// Return given quadrature weight
auto GridStructure::get_weights( int nN ) const -> Real {
  return Weights( nN );
}

// Acessor for xL
auto GridStructure::get_x_l( ) const -> Real { return xL; }

// Acessor for xR
auto GridStructure::get_x_r( ) const -> Real { return xR; }

// Acessor for SqrtGm
auto GridStructure::get_sqrt_gm( Real X ) const -> Real {
  if ( Geometry == geometry::Spherical ) {
    return X * X;
  }
  return 1.0;
}

// Accessor for X_L
auto GridStructure::get_left_interface( int iX ) const -> Real {
  return X_L( iX );
}

// Return nNodes
auto GridStructure::get_n_nodes( ) const noexcept -> int { return nNodes; }

// Return nElements
auto GridStructure::get_n_elements( ) const noexcept -> int {
  return nElements;
}

// Return number of guard zones
auto GridStructure::get_guard( ) const noexcept -> int { return nGhost; }

// Return first physical zone
auto GridStructure::get_ilo( ) const noexcept -> int { return nGhost; }

// Return last physical zone
auto GridStructure::get_ihi( ) const noexcept -> int {
  return nElements + nGhost - 1;
}

// Return true if in spherical symmetry
auto GridStructure::do_geometry( ) const noexcept -> bool {
  return Geometry == geometry::Spherical;
}

// Equidistant mesh
// TODO(astrobarker): We will need to replace Centers here, right?
void GridStructure::create_grid( ) const {

  const int ilo = nGhost; // first real zone
  const int ihi = nElements + nGhost - 1; // last real zone

  for ( int i = 0; i < nElements + 2 * nGhost; i++ ) {
    Widths( i ) = ( xR - xL ) / nElements;
  }

  X_L( nGhost ) = xL;
  for ( int iX = 2; iX < nElements + 2 * nGhost; iX++ ) {
    X_L( iX ) = X_L( iX - 1 ) + Widths( iX - 1 );
  }

  Centers( ilo ) = xL + 0.5 * Widths( ilo );
  for ( int i = ilo + 1; i <= ihi; i++ ) {
    Centers( i ) = Centers( i - 1 ) + Widths( i - 1 );
  }

  for ( int i = ilo - 1; i >= 0; i-- ) {
    Centers( i ) = Centers( i + 1 ) - Widths( i + 1 );
  }
  for ( int i = ihi + 1; i < nElements + nGhost + 1; i++ ) {
    Centers( i ) = Centers( i - 1 ) + Widths( i - 1 );
  }

  for ( int iC = ilo; iC <= ihi; iC++ ) {
    for ( int iN = 0; iN < nNodes; iN++ ) {
      Grid( iC, iN ) = node_coordinate( iC, iN );
    }
  }
}

/**
 * Compute cell masses
 **/
void GridStructure::compute_mass( View3D<Real> uPF ) {
  const int nNodes = get_n_nodes( );
  const int ilo    = get_ilo( );
  const int ihi    = get_ihi( );

  Real mass = NAN;
  Real X    = NAN;

  for ( int iX = ilo; iX <= ihi; iX++ ) {
    mass = 0.0;
    for ( int iN = 0; iN < nNodes; iN++ ) {
      X = node_coordinate( iX, iN );
      mass += Weights( iN ) * get_sqrt_gm( X ) * uPF( 0, iX, iN );
    }
    mass *= Widths( iX );
    Mass( iX ) = mass;
  }

  // Guard cells
  for ( int iX = 0; iX < ilo; iX++ ) {
    Mass( ilo - 1 - iX ) = Mass( ilo + iX );
    Mass( ihi + 1 + iX ) = Mass( ihi - iX );
  }
}

/**
 * Compute cell centers of masses reference coordinates
 **/
void GridStructure::compute_center_of_mass( View3D<Real> uPF ) {
  const int nNodes = get_n_nodes( );
  const int ilo    = get_ilo( );
  const int ihi    = get_ihi( );

  Real com = NAN;
  Real X   = NAN;

  for ( int iX = ilo; iX <= ihi; iX++ ) {
    com = 0.0;
    for ( int iN = 0; iN < nNodes; iN++ ) {
      X = node_coordinate( iX, iN );
      com += Nodes( iN ) * Weights( iN ) * get_sqrt_gm( X ) * uPF( 0, iX, iN );
    }
    com *= Widths( iX );
    CenterOfMass( iX ) = com / Mass( iX );
  }

  // Guard cells
  for ( int iX = 0; iX < ilo; iX++ ) {
    CenterOfMass( ilo - 1 - iX ) = CenterOfMass( ilo + iX );
    CenterOfMass( ihi + 1 + iX ) = CenterOfMass( ihi - iX );
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
        X_L( iX )     = SData( iX );
        Widths( iX )  = SData( iX + 1 ) - SData( iX );
        Centers( iX ) = 0.5 * ( SData( iX + 1 ) + SData( iX ) );
      } );

  Kokkos::parallel_for(
      "Grid Update 2", Kokkos::RangePolicy<>( ilo, ihi + 2 ),
      KOKKOS_CLASS_LAMBDA( int iX ) {
        for ( int iN = 0; iN < nNodes; iN++ ) {
          Grid( iX, iN ) = node_coordinate( iX, iN );
        }
      } );
}

// Access by (element, node)
auto GridStructure::operator( )( int i, int j ) -> Real& {
  return Grid( i, j );
}

auto GridStructure::operator( )( int i, int j ) const -> Real {
  return Grid( i, j );
}
