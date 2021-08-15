/**
 * Class for holding the grid data.
 * For a loop over real zones, loop from ilo to ihi (inclusive).
 * ilo = nGhost
 * ihi = nElements - nGhost + 1

 TODO: Write GetNodes, GetWeights(nN) functions
 **/

#include <iostream>
#include "QuadratureLibrary.h"

class GridStructure
{
public:

  // default constructor - for move constructur
  GridStructure()
  {
  nElements = 1;
  nNodes    = 1;
  nGhost    = 1;

  mSize = nElements + 2 * nGhost;

  xL = 0.0;
  xR = 1.0;

  // Compute quadrature weights and nodes
  Nodes   = new double[ nNodes ];
  Weights = new double[ nNodes ];

  LG_Quadrature( nNodes, Nodes, Weights );

  Centers = new double[mSize];
  Widths  = new double[mSize];
  Grid    = new double[(mSize) * nNodes];
  CreateGrid();
  }

  // copy-constructor
  //TODO: Need to extend these implementations to 
  //   include Centers, Widths, etc
  //https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
  GridStructure(const GridStructure& other)
      : nElements(other.nElements),
        nNodes(other.nNodes),
        nGhost(other.nGhost),
        mSize(other.mSize),
        xL(other.xL),
        xR(other.xR),
        Nodes(nNodes ? new double[nNodes] : nullptr),
        Weights(nNodes ? new double[nNodes] : nullptr),
        Centers(mSize ? new double[mSize] : nullptr),
        Widths(mSize ? new double[mSize] : nullptr),
        Grid(mSize&&nNodes ? new double[mSize*nNodes] : nullptr)
  {
      // note that this is non-throwing, because of the data
      // types being used; more attention to detail with regards
      // to exceptions must be given in a more general case, however
      std::copy( other.Centers, other.Centers + mSize, Centers );
      std::copy( other.Widths, other.Widths + mSize, Widths );
      std::copy( other.Grid, other.Grid + mSize*nNodes, Grid );
      std::copy( other.Nodes, other.Nodes + nNodes, Nodes );
      std::copy( other.Weights, other.Weights + nNodes, Weights );
  }

  friend void swap(GridStructure& first, GridStructure& second) // nothrow
    {
      // by swapping the members of two objects,
      // the two objects are effectively swapped
      std::swap( first.nElements, second.nElements );
      std::swap( first.nNodes, second.nNodes );
      std::swap( first.nGhost, second.nGhost );
      std::swap( first.mSize, second.mSize );
      std::swap( first.xL, second.xL );
      std::swap( first.xR, second.xR );
      std::swap( first.Grid, second.Grid );
      std::swap( first.Centers, second.Centers );
      std::swap( first.Widths, second.Widths );
    }

  // assignment
  GridStructure& operator=(GridStructure other) 
  {
    swap(*this, other);

    return *this;
  }

  // move constructor
  GridStructure(GridStructure&& other) noexcept
    : GridStructure() // initialize via default constructor, C++11 only
  {
    swap(*this, other);
  }

  GridStructure( unsigned int nX, unsigned int nN, unsigned int nG, 
  double left, double right );
  double NodeCoordinate( unsigned int iC, unsigned int iN );
  double& operator()( unsigned int i, unsigned int j );
  double operator()( unsigned int i, unsigned int j ) const;
  void CreateGrid( );

  ~GridStructure()
  {
    delete [] Nodes;
    delete [] Weights;
    delete [] Centers;
    delete [] Widths;
    delete [] Grid;
  }

private:
  unsigned int nElements;
  unsigned int nNodes;
  unsigned int nGhost;
  unsigned int mSize;

  double xL;
  double xR;

  double* Nodes;
  double* Weights;

  double* Centers;
  double* Widths;
  double* Grid;
};

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