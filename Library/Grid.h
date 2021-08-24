#ifndef GRID_H
#define GRID_H

/**
 * Class for holding the grid data.
 * For a loop over real zones, loop from ilo to ihi (inclusive).
 * ilo = nGhost
 * ihi = nElements - nGhost + 1

 TODO: Write GetNodes, GetWeights(nN) functions
**/

#include "QuadratureLibrary.h"
#include <algorithm>    // std::copy
#include <vector>

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
  double Get_Widths( unsigned int iC );
  double Get_Nodes( unsigned int nN );
  double Get_Weights( unsigned int nN ); 
  int Get_Guard( );
  int Get_ilo( );
  int Get_ihi( );
  int Get_nNodes( );
  int Get_nElements( );

  double& operator()( unsigned int i, unsigned int j );
  double operator()( unsigned int i, unsigned int j ) const;
  void copy( std::vector<double> dest );
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

#endif
