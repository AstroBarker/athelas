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


#include "Error.h"
#include "QuadratureLibrary.h"

class GridStructure
{
 public:
  GridStructure( unsigned int nN, unsigned int nX, unsigned int nG, double left,
                 double right, bool Geom );
  double NodeCoordinate( unsigned int iC, unsigned int iN ) const;
  double Get_Centers( unsigned int iC ) const;
  double Get_Widths( unsigned int iC ) const;
  double Get_Nodes( unsigned int nN ) const;
  double Get_Weights( unsigned int nN ) const;
  double Get_Mass( unsigned int iX ) const;
  double Get_CenterOfMass( unsigned int iX ) const;
  double Get_xL( ) const;
  double Get_xR( ) const;
  double Get_SqrtGm( double X ) const;
  double Get_LeftInterface( unsigned int iX ) const;

  bool DoGeometry( ) const;

  int Get_Guard( ) const;
  int Get_ilo( ) const;
  int Get_ihi( ) const;
  int Get_nNodes( ) const;
  int Get_nElements( ) const;

  void CreateGrid( );
  void UpdateGrid( Kokkos::View<double*> SData );
  void ComputeMass( Kokkos::View<double***> uPF );
  void ComputeCenterOfMass( Kokkos::View<double***> uPF );
  void ComputeCenterOfMass_Radius( Kokkos::View<double***> uPF );

  double& operator( )( unsigned int i, unsigned int j );
  double operator( )( unsigned int i, unsigned int j ) const;

 private:
  unsigned int nElements;
  unsigned int nNodes;
  unsigned int nGhost;
  unsigned int mSize;

  double xL;
  double xR;

  bool Geometry;

  Kokkos::View<double*> Nodes;
  Kokkos::View<double*> Weights;

  Kokkos::View<double*> Centers;
  Kokkos::View<double*> Widths;
  Kokkos::View<double*> X_L; // left interface coordinate

  Kokkos::View<double*> Mass;
  Kokkos::View<double*> CenterOfMass;

  Kokkos::View<double**> Grid;
};

#endif
