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
 * TODO: Convert Grid to vectors.
**/ 

#include "QuadratureLibrary.h"
#include "Grid.h"
#include "Error.h"
#include  <iostream>

GridStructure::GridStructure( unsigned int nN, unsigned int nX, unsigned int nG, double left, double right )
{
  nElements = nX;
  nNodes = nN;
  nGhost = nG;

  mSize = nElements + 2 * nGhost;

  xL = left;
  xR = right;

  // Compute quadrature weights and nodes
  Nodes   = new double[ nNodes ];
  Weights = new double[ nNodes ];

  LG_Quadrature( nNodes, Nodes, Weights );

  Centers = new double[mSize];
  Widths  = new double[mSize];
  Grid = new double[(mSize) * nNodes];
  CreateGrid();
}


// Give physical grid coordinate from a node.
double GridStructure::NodeCoordinate( unsigned int iC, unsigned int iN )
{
  return Centers[iC] + Widths[iC] * Nodes[iN];
}


// Update nodal coordinates with nodal velocities
// void GridStructure::UpdateGrid( double vel )


// Return cell center
double GridStructure::Get_Centers( unsigned int iC )
{
  return Centers[iC];
}


// Return cell width
double GridStructure::Get_Widths( unsigned int iC )
{
  return Widths[iC];
}


// Return cell Volume
double GridStructure::Get_Volume( unsigned int iX )
{
  return Volume[iX];
}


// Return cell mass
double GridStructure::Get_Mass( unsigned int iX )
{
  return Mass[iX];
}


// Return cell Volume
double GridStructure::Get_CenterOfMass( unsigned int iX )
{
  return CenterOfMass[iX];
}


// Return given quadrature node
double GridStructure::Get_Nodes( unsigned int nN )
{
  return Nodes[nN];
}


// Return given quadrature weight
double GridStructure::Get_Weights( unsigned int nN )
{
  return Weights[nN];
}


// Return nNodes
int GridStructure::Get_nNodes( )
{
  return nNodes;
}


// Return nElements
int GridStructure::Get_nElements( )
{
  return nElements;
}


// Return number of guard zones
int GridStructure::Get_Guard( )
{
  return nGhost;
}


// Return first physical zone
int GridStructure::Get_ilo( )
{
  return nGhost;
}


// Return last physical zone
int GridStructure::Get_ihi( )
{
  return nElements + nGhost - 1;
}


// Copy Grid contents into new array
// Not elegant.. as the copy shouldn't include guard cells.
void GridStructure::copy( std::vector<double> dest )
{

  unsigned int j;
  for ( unsigned int i = nGhost*nNodes; i <= (nElements + nGhost - 1)*nNodes + 1; i++ )
  {
    j = i - nGhost*nNodes;
    dest[j] = Grid[i];
  }

}


// Equidistant mesh
void GridStructure::CreateGrid( )
{

  const unsigned int ilo = nGhost; // first real zone
  const unsigned int ihi = nElements - nGhost + 1; // last real zone

  for (unsigned int i = 0; i < nElements + 2 * nGhost; i++)
  {
    Widths[i] = ( xR - xL ) / nElements;
  }

  Centers[ilo] = xL + 0.5 * Widths[ilo];
  for (unsigned int i = ilo+1; i <= ihi; i++)
  {
    Centers[i] = Centers[i-1] + Widths[i-1];
  }

  for (int i = ilo-1; i >= 0; i--)
  {
    Centers[i] = Centers[i+1] - Widths[i+1];
  }
  for (unsigned int i = ihi+1; i < nElements + nGhost + 1; i++)
  {
    Centers[i] = Centers[i-1] + Widths[i-1];
  }

  for (unsigned int iC = ilo; iC <= ihi; iC++)
  {
    for (unsigned int iN = 0; iN < nNodes; iN++)
    {
      Grid[iC * nNodes + iN] = NodeCoordinate( iC, iN );
    }
  }

}

// Return center of given cell
double GridStructure::CellAverage( unsigned int iX )
{

  double avg = 0.0;

  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    avg += Weights[iN] * Grid[iX * nNodes + iN];
  }

  return avg;
}


/**
 * Compute cell volumes element (e.g., 4 pi r^2 dr)
**/
void GridStructure::ComputeVolume(  )
{
  const unsigned int nNodes = Get_nNodes();
  const unsigned int ilo    = Get_ilo();
  const unsigned int ihi    = Get_ihi();

  double geom = 1.0; // Temporary
  double vol;

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    vol = 0.0;
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      vol += geom * Get_Widths(iX) * Weights[iN];
    }
    Volume[iX] = vol;
  }
}


/**
 * Compute cell masses
**/
void GridStructure::ComputeMass( DataStructure3D& uPF )
{
  const unsigned int nNodes = Get_nNodes();
  const unsigned int ilo    = Get_ilo();
  const unsigned int ihi    = Get_ihi();

  double mass;

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    mass = 0.0;
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      mass += uPF(0,iX,iN) * Volume[iX] * Weights[iN];
    }
    Mass[iX] = mass;
  }
}


/**
 * Compute cell centers of masses
**/
void GridStructure::ComputeCenterOfMass( DataStructure3D& uPF )
{
  const unsigned int nNodes = Get_nNodes();
  const unsigned int ilo    = Get_ilo();
  const unsigned int ihi    = Get_ihi();

  double com;

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    com = 0.0;
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      com += uPF(0,iX,iN) * Volume[iX] * Nodes[iN] * Weights[iN];
    }
    CenterOfMass[iX] = com / Mass[iX];
  }
}


/**
 * Update grid coordinates using interface and nodal velocities.
**/
void GridStructure::UpdateGrid( DataStructure3D& U,
                 std::vector<double>& Flux_U, double dt )
{

  const unsigned int nNodes = Get_nNodes();
  const unsigned int ilo    = Get_ilo();
  const unsigned int ihi    = Get_ihi();

  double dx_L = 0.0; // Left interfact of element
  double dx_R = 0.0; // Right interface of element
  double dx   = 0.0; // Cumulative change

  double Vel_Avg = 0.0;
  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {

    // --- Update Cell Widths ---

    if ( Flux_U[iX]+1.0 == 1.0 && Flux_U[iX+1]+1.0 == 1.0 ) 
    {
      continue;
    }

    dx_L = + Flux_U[iX] * dt;
    dx_R = + Flux_U[iX+1] * dt; // TODO: Make sure this isn't accessing bad data
    // std::printf("%d %.25f %.25f %.7f %.7f \n", iX, Flux_U[iX]+0.0, Flux_U[iX+1]+0.0, dx_L, dx_R);
    // Combine the changes in interface coordinates
    // Left compression, right expansion
    if ( dx_L >= 0.0 && dx_R >= 0.0 )
    {
      dx = dx_R - dx_L;
    }
    // Left expasion, right compression
    else if ( dx_L <= 0.0 && dx_R <= 0.0 )
    {
      dx = std::abs(dx_L) - std::abs(dx_R);
    }
    // dual expansion
    else if ( dx_L <= 0.0 && dx_R >= 0.0 )
    {
      dx = std::abs(dx_L) + dx_R;
    }
    // dual compression
    else if ( dx_L >= 0.0 && dx_R <= 0.0 )
    {
      dx = dx_R - dx_L;
    }
    // Unknown behavior, throw an error. Shouldn't occur.
    else
    {
      std::printf(" V_L, V_R, dx_L, dx_R: %.5e %.5e %.5e %.5e\n", 
        Flux_U[iX], Flux_U[iX+1], dx_L, dx_R);
      std::printf("Cell: %d\n", iX);
      throw Error("Unknown behavior encountered in Grid Update.");
    }

    // Given dx, update the current cell width.
    Widths[iX] += dx;

    // --- Update Nodal Values ---

    std::vector<double> Weights(nNodes);
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      Weights[iN] = Get_Weights( iN );
    }

    Vel_Avg = U.CellAverage( 1, iX, nNodes, Weights );

    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      // grid update -- nodal
      // Grid[iX * nNodes + iN] += Vel_Avg * dt;     
      Grid[iX * nNodes + iN] += Vel_Avg * dt;      
    }

    // --- Update cell centers ---
    Centers[iX] += Vel_Avg * dt; // = CellAverage( iX );

  }
}

// Access by (element, node)
double& GridStructure::operator()( unsigned int i, unsigned int j )
{
  return Grid[i * nNodes + j];
}

double GridStructure::operator()( unsigned int i, unsigned int j ) const
{
  return Grid[i * nNodes + j];
}