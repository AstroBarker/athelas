/**
 * File     :  Grid.h
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the grid data.
 *  For a loop over real zones, loop from ilo to ihi (inclusive).
 *  ilo = nGhost
 *  ihi = nElements - nGhost + 1
 *
 * TODO: Can we initialize a Grid object from another?
 **/

#include "Grid.h"
#include "Constants.h"
#include <math.h> /* atan */

GridStructure::GridStructure( unsigned int nN, unsigned int nX, unsigned int nG,
                              double left, double right, bool Geom )
    : nElements( nX ), nNodes( nN ), nGhost( nG ),
      mSize( nElements + 2 * nGhost ), xL( left ), xR( right ),
      Geometry( Geom ), Nodes( nNodes ), Weights( nNodes ),
      Centers( mSize, 0.0 ), Widths( mSize, 0.0 ), X_L( mSize, 0.0 ),
      Mass( mSize, 0.0 ), CenterOfMass( mSize, 0.0 ),
      Grid( mSize * nNodes, 0.0 )
{
  // TODO: Allow LG_Quadrature to take in vectors.
  double* tmp_nodes   = new double[nNodes];
  double* tmp_weights = new double[nNodes];

  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    tmp_nodes[iN]   = 0.0;
    tmp_weights[iN] = 0.0;
  }

  LG_Quadrature( nNodes, tmp_nodes, tmp_weights );

  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    Nodes[iN]   = tmp_nodes[iN];
    Weights[iN] = tmp_weights[iN];
  }

  CreateGrid( );

  delete[] tmp_nodes;
  delete[] tmp_weights;
}

// linear shape function on the reference element
double ShapeFunction( int interface, double eta )
{
  double mult = 1.0;

  if ( interface == 0 ) mult = -1.0;
  if ( interface == 1 ) mult = +1.0;
  if ( interface != 0 && interface != 1 )
    throw Error( "Invalid shape func params" );

  return 0.5 + mult * eta;
}

// Give physical grid coordinate from a node.
double GridStructure::NodeCoordinate( unsigned int iC, unsigned int iN )
{
  return X_L[iC] * ShapeFunction( 0, Nodes[iN] ) +
         X_L[iC + 1] * ShapeFunction( 1, Nodes[iN] );
}

// Return cell center
double GridStructure::Get_Centers( unsigned int iC ) { return Centers[iC]; }

// Return cell width
double GridStructure::Get_Widths( unsigned int iC ) { return Widths[iC]; }

// Return cell mass
double GridStructure::Get_Mass( unsigned int iX ) { return Mass[iX]; }

// Return cell reference Center of Mass
double GridStructure::Get_CenterOfMass( unsigned int iX )
{
  return CenterOfMass[iX];
}

// Return given quadrature node
double GridStructure::Get_Nodes( unsigned int nN ) { return Nodes[nN]; }

// Return given quadrature weight
double GridStructure::Get_Weights( unsigned int nN ) { return Weights[nN]; }

// Acessor for xL
double GridStructure::Get_xL( ) { return xL; }

// Acessor for xR
double GridStructure::Get_xR( ) { return xR; }

// Acessor for SqrtGm
double GridStructure::Get_SqrtGm( double X )
{
  if ( Geometry )
  {
    // return 4.0 * PI() * X * X;
    return X * X;
  }
  else
  {
    return 1.0;
  }
}

// Accessor for X_L
double GridStructure::Get_LeftInterface( unsigned int iX ) { return X_L[iX]; }

// Return nNodes
int GridStructure::Get_nNodes( ) { return nNodes; }

// Return nElements
int GridStructure::Get_nElements( ) { return nElements; }

// Return number of guard zones
int GridStructure::Get_Guard( ) { return nGhost; }

// Return first physical zone
int GridStructure::Get_ilo( ) { return nGhost; }

// Return last physical zone
int GridStructure::Get_ihi( ) { return nElements + nGhost - 1; }

// Return true if in spherical symmetry
bool GridStructure::DoGeometry( ) { return Geometry; }

// Equidistant mesh
// TODO: We will need to replace Centers here, right?
void GridStructure::CreateGrid( )
{

  const unsigned int ilo = nGhost;                 // first real zone
  const unsigned int ihi = nElements + nGhost - 1; // last real zone

  for ( unsigned int i = 0; i < nElements + 2 * nGhost; i++ )
  {
    Widths[i] = ( xR - xL ) / nElements;
  }

  X_L[nGhost] = xL;
  for ( unsigned int iX = 2; iX < nElements + 2 * nGhost; iX++ )
  {
    X_L[iX] = X_L[iX - 1] + Widths[iX - 1];
  }

  Centers[ilo] = xL + 0.5 * Widths[ilo];
  for ( unsigned int i = ilo + 1; i <= ihi; i++ )
  {
    Centers[i] = Centers[i - 1] + Widths[i - 1];
  }

  for ( int i = ilo - 1; i >= 0; i-- )
  {
    Centers[i] = Centers[i + 1] - Widths[i + 1];
  }
  for ( unsigned int i = ihi + 1; i < nElements + nGhost + 1; i++ )
  {
    Centers[i] = Centers[i - 1] + Widths[i - 1];
  }

  for ( unsigned int iC = ilo; iC <= ihi; iC++ )
  {
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      Grid[iC * nNodes + iN] = NodeCoordinate( iC, iN );
    }
  }
}

/**
 * Compute cell masses
 **/
void GridStructure::ComputeMass( DataStructure3D& uPF )
{
  const unsigned int nNodes = Get_nNodes( );
  const unsigned int ilo    = Get_ilo( );
  const unsigned int ihi    = Get_ihi( );

  double mass;
  double X;

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    mass = 0.0;
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      X = NodeCoordinate( iX, iN );
      mass += Weights[iN] * Get_SqrtGm( X ) * uPF( 0, iX, iN );
    }
    mass *= Widths[iX];
    Mass[iX] = mass;
  }

  // Guard cells
  for ( unsigned int iX = 0; iX < ilo; iX++ )
  {
    Mass[ilo - 1 - iX] = Mass[ilo + iX];
    Mass[ihi + 1 + iX] = Mass[ihi - iX];
  }
}

/**
 * Compute cell centers of masses reference coordinates
 **/
void GridStructure::ComputeCenterOfMass( DataStructure3D& uPF )
{
  const unsigned int nNodes = Get_nNodes( );
  const unsigned int ilo    = Get_ilo( );
  const unsigned int ihi    = Get_ihi( );

  double com;
  double X;

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    com = 0.0;
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      X = NodeCoordinate( iX, iN );
      com += Nodes[iN] * Weights[iN] * Get_SqrtGm( X ) * uPF( 0, iX, iN );
    }
    com *= Widths[iX];
    CenterOfMass[iX] = com / Mass[iX];
  }

  // Guard cells
  for ( unsigned int iX = 0; iX < ilo; iX++ )
  {
    CenterOfMass[ilo - 1 - iX] = CenterOfMass[ilo + iX];
    CenterOfMass[ihi + 1 + iX] = CenterOfMass[ihi - iX];
  }
}

/**
 * Update grid coordinates using interface velocities.
 **/
void GridStructure::UpdateGrid( std::vector<double>& SData )
{

  const unsigned int ilo = Get_ilo( );
  const unsigned int ihi = Get_ihi( );

  for ( unsigned int iX = ilo; iX <= ihi + 1; iX++ )
  {
    X_L[iX]     = SData[iX];
    Widths[iX]  = SData[iX + 1] - SData[iX];
    Centers[iX] = 0.5 * ( SData[iX + 1] + SData[iX] );
  }

  for ( unsigned int iC = ilo; iC <= ihi; iC++ )
  {
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      Grid[iC * nNodes + iN] = NodeCoordinate( iC, iN );
    }
  }
}

// Access by (element, node)
double& GridStructure::operator( )( unsigned int i, unsigned int j )
{
  return Grid[i * nNodes + j];
}

double GridStructure::operator( )( unsigned int i, unsigned int j ) const
{
  return Grid[i * nNodes + j];
}