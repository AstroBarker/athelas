#ifndef GRID_H
#define GRID_H

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
 **/

#include <algorithm> // std::copy
#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Error.h"
#include "QuadratureLibrary.h"

class GridStructure
{
 public:
  GridStructure( unsigned int nN, unsigned int nX, unsigned int nG, Real left,
                 Real right, bool Geom );
  Real NodeCoordinate( unsigned int iC, unsigned int iN ) const;
  Real Get_Centers( unsigned int iC ) const;
  Real Get_Widths( unsigned int iC ) const;
  Real Get_Nodes( unsigned int nN ) const;
  Real Get_Weights( unsigned int nN ) const;
  Real Get_Mass( unsigned int iX ) const;
  Real Get_CenterOfMass( unsigned int iX ) const;
  Real Get_xL( ) const;
  Real Get_xR( ) const;
  Real Get_SqrtGm( Real X ) const;
  Real Get_LeftInterface( unsigned int iX ) const;

  bool DoGeometry( ) const;

  int Get_Guard( ) const;
  int Get_ilo( ) const;
  int Get_ihi( ) const;
  int Get_nNodes( ) const;
  int Get_nElements( ) const;

  void CreateGrid( );
  void UpdateGrid( Kokkos::View<Real*> SData );
  void ComputeMass( Kokkos::View<Real***> uPF );
  void ComputeCenterOfMass( Kokkos::View<Real***> uPF );
  void ComputeCenterOfMass_Radius( Kokkos::View<Real***> uPF );

  Real& operator( )( unsigned int i, unsigned int j );
  Real operator( )( unsigned int i, unsigned int j ) const;

 private:
  unsigned int nElements;
  unsigned int nNodes;
  unsigned int nGhost;
  unsigned int mSize;

  Real xL;
  Real xR;

  bool Geometry;

  Kokkos::View<Real*> Nodes;
  Kokkos::View<Real*> Weights;

  Kokkos::View<Real*> Centers;
  Kokkos::View<Real*> Widths;
  Kokkos::View<Real*> X_L; // left interface coordinate

  Kokkos::View<Real*> Mass;
  Kokkos::View<Real*> CenterOfMass;

  Kokkos::View<Real**> Grid;
};

#endif
