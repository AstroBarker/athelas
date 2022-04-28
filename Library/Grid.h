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

#include "DataStructures.h"
#include "Error.h"
#include "QuadratureLibrary.h"

class GridStructure
{
 public:
  GridStructure( unsigned int nN, unsigned int nX, unsigned int nG, double left,
                 double right, bool Geom );
  double NodeCoordinate( unsigned int iC, unsigned int iN );
  double Get_Centers( unsigned int iC );
  double Get_Widths( unsigned int iC );
  double Get_Nodes( unsigned int nN );
  double Get_Weights( unsigned int nN );
  double Get_Mass( unsigned int iX );
  double Get_CenterOfMass( unsigned int iX );
  double Get_xL( );
  double Get_xR( );
  double Get_SqrtGm( double X );
  double Get_LeftInterface( unsigned int iX );

  bool DoGeometry( );

  int Get_Guard( );
  int Get_ilo( );
  int Get_ihi( );
  int Get_nNodes( );
  int Get_nElements( );

  void CreateGrid( );
  void UpdateGrid( std::vector<double>& SData );
  void ComputeMass( DataStructure3D& uPF );
  void ComputeCenterOfMass( DataStructure3D& uPF );
  void ComputeCenterOfMass_Radius( DataStructure3D& uPF );

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

  std::vector<double> Nodes;
  std::vector<double> Weights;

  std::vector<double> Centers;
  std::vector<double> Widths;
  std::vector<double> X_L; // left interface coordinate

  std::vector<double> Mass;
  std::vector<double> CenterOfMass;

  std::vector<double> Grid;
};

#endif
