/**
 * Class for holding the grid data.
 * For a loop over real zones, loop from ilo to ihi (inclusive).
 * ilo = nGhost
 * ihi = nElements - nGhost + 1

 TODO: Write GetNodes, GetWeights(nN) functions
**/

#include "QuadratureLibrary.h"
#include "Grid.h"

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
  return nElements - nGhost + 1;
}

// Copy Grid contents into new array
void GridStructure::copy( double* dest )
{

  // for ( unsigned int i = 0; i < mSize*nNodes; i++ )
  // {
    // dest[i] = Grid[i];
  // }
  for ( unsigned int ix = 0; ix < mSize; ix++ )
  for ( unsigned int in = 0; in < nNodes; in++ )
  {
    dest[in * nElements + ix] = Grid[in * nElements + ix];
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
      Grid[iN * nElements + iC] = NodeCoordinate( iC, iN );
    }
  }

}

// Access by (node, element)
double& GridStructure::operator()( unsigned int i, unsigned int j )
{
  return Grid[i * nElements + j];
}

double GridStructure::operator()( unsigned int i, unsigned int j ) const
{
  return Grid[i * nElements + j];
}