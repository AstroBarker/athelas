/**
 * Class for holding the grid data.
 * For a loop over real zones, loop from ilo to ihi (inclusive).
 * ilo = nGhost
 * ihi = nElements - nGhost + 1
 **/

#include <iostream>
#include "QuadratureLibrary.h"

class GridStructure
{
public:
  GridStructure( unsigned int nX, unsigned int nN, unsigned int nG, 
  double left, double right );
  double NodeCoordinate( unsigned int iC, unsigned int iN );
  double& operator()( unsigned int i, unsigned int j );
  double operator()( unsigned int i, unsigned int j ) const;
  void CreateGrid( );

  double* Nodes;
  double* Weights;

  ~GridStructure()
  {
    delete [] Grid;
    delete [] Centers;
    delete [] Widths;
    delete [] Nodes;
    delete [] Weights;
  }

private:
  unsigned int nElements;
  unsigned int nNodes;
  unsigned int nGhost;

  double* Grid;
  double* Centers;
  double* Widths;

  double xL;
  double xR;
};

GridStructure::GridStructure( unsigned int nN, unsigned int nX, unsigned int nG, double left, double right )
{
  nElements = nX;
  nNodes = nN;
  nGhost = nG;

  xL = left;
  xR = right;

  // Compute quadrature weights and nodes
  Nodes   = new double[ nNodes ];
  Weights = new double[ nNodes ];

  LG_Quadrature( nNodes, Nodes, Weights );

  Grid = new double[(nElements + 2*nGhost) * nNodes];
  Centers = new double[nElements + 2 * nGhost];
  Widths  = new double[nElements + 2 * nGhost];
  CreateGrid();
}

// Give physical grid coordinate from a node.
double GridStructure::NodeCoordinate( unsigned int iC, unsigned int iN )
{
  return Centers[iC] + Widths[iC] * Nodes[iN];
}

// Equidistant mesh
void GridStructure::CreateGrid( )
{

  unsigned int ilo = nGhost; // first real zone
  unsigned int ihi = nElements - nGhost + 1; // last real zone

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