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
 * TODO: Convert Grid to vectors.
**/ 

#include <iostream>
#include <algorithm>    // std::copy
#include <vector>

#include "QuadratureLibrary.h"
#include "DataStructures.h"
#include "Error.h"

class GridStructure
{
public:

  GridStructure( unsigned int nX, unsigned int nN, unsigned int nS,
    unsigned int nG, double left, double right );
  double NodeCoordinate( unsigned int iC, unsigned int iN ); // TODO: NodeCoordinate needs updating for modal
  double Get_Centers( unsigned int iC );
  double Get_Widths( unsigned int iC );
  double Get_Nodes( unsigned int nN );
  double Get_Weights( unsigned int nN ); 
  double Get_Volume( unsigned int iX );
  double Get_Mass( unsigned int iX );
  double Get_CenterOfMass( unsigned int iX );
  
  int Get_Guard( );
  int Get_ilo( );
  int Get_ihi( );
  int Get_nNodes( );
  int Get_nElements( );

  double& operator()( unsigned int i, unsigned int j );
  double operator()( unsigned int i, unsigned int j ) const;
  void CreateGrid( );
  double CellAverage( unsigned int iX );
  void UpdateGrid( std::vector<double>& SData );
  void ComputeMass( DataStructure3D& uPF );
  void ComputeVolume(  );
  void ComputeCenterOfMass( DataStructure3D& uPF );

private:
  unsigned int nElements;
  unsigned int nNodes;
  unsigned int nGhost;
  unsigned int mSize;

  double xL;
  double xR;

  std::vector<double> Nodes;
  std::vector<double> Weights;

  std::vector<double> Centers;
  std::vector<double> Widths;
  std::vector<double> Work;

  std::vector<double> Mass;
  std::vector<double> Volume;
  std::vector<double> CenterOfMass;

  std::vector<std::vector<double>> StageData;

  std::vector<double> Grid;

};

#endif
