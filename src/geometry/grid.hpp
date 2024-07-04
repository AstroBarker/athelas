#ifndef GRID_HPP_
#define GRID_HPP_

/**
 * File     :  grid.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the grid data.
 *  For a loop over real zones, loop from ilo to ihi (inclusive).
 *  ilo = nGhost
 *  ihi = nElements - nGhost + 1
 *
 **/

#include <algorithm> // std::copy
#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "error.hpp"
#include "geometry.hpp"
#include "problem_in.hpp"
#include "quadrature.hpp"

class GridStructure {
 public:
  GridStructure( ProblemIn *pin );
  Real NodeCoordinate( int iC, int iN ) const;
  Real Get_Centers( int iC ) const;
  Real Get_Widths( int iC ) const;
  Real Get_Nodes( int nN ) const;
  Real Get_Weights( int nN ) const;
  Real Get_Mass( int iX ) const;
  Real Get_CenterOfMass( int iX ) const;
  Real Get_xL( ) const;
  Real Get_xR( ) const;
  Real Get_SqrtGm( Real X ) const;
  Real Get_LeftInterface( int iX ) const;

  bool DoGeometry( ) const;

  int Get_Guard( ) const;
  int Get_ilo( ) const;
  int Get_ihi( ) const;
  int Get_nNodes( ) const;
  int Get_nElements( ) const;

  void CreateGrid( );
  void UpdateGrid( View1D<Real> SData );
  void ComputeMass( View3D<Real> uPF );
  void ComputeCenterOfMass( View3D<Real> uPF );
  void ComputeCenterOfMass_Radius( View3D<Real> uPF );

  Real &operator( )( int i, int j );
  Real operator( )( int i, int j ) const;

 private:
  int nElements;
  int nNodes;
  int nGhost;
  int mSize;

  Real xL;
  Real xR;

  geometry::Geometry Geometry;

  View1D<Real> Nodes;
  View1D<Real> Weights;

  View1D<Real> Centers;
  View1D<Real> Widths;
  View1D<Real> X_L; // left interface coordinate

  View1D<Real> Mass;
  View1D<Real> CenterOfMass;

  View2D<Real> Grid;
};

#endif // GRID_HPP_
