/**
 * File     :  DataStructures.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Classes for holding multidimensional data.
 *  Multi-D structures are wrapped around 1D vectors to conveniently and 
 *  efficiently access data. For conserved variables data structure, 
 *  initialize/acces as DataStructures3D uCF(nCF,nX,order)
 * Contains : DataStructures2D, DataStructures3D
**/ 

#include <vector>
#include "DataStructures.h"

DataStructure2D::DataStructure2D( unsigned int rows, unsigned int cols )
  : Rows(rows),
    Cols(cols),
    mSize(Rows * Cols),
    Data(mSize, 0.0)
{
}

//TODO: When we use this, make sure it is accessing data efficiently.
double& DataStructure2D::operator()( unsigned int i, unsigned int j )
{
  return Data[i * Cols + j];
}

double DataStructure2D::operator()( unsigned int i, unsigned int j ) const
{
  return Data[i * Cols + j];
}

// init as {nCF, nX, order}
DataStructure3D::DataStructure3D( unsigned int N1, unsigned int N2, unsigned int N3 )
  : Size1(N1),
    Size2(N2),
    Size3(N3),
    mSize(Size1*Size2*Size2),
    Data(mSize, 0.0)
{
}

// access (iCF, iX, iN)
//TODO: DataStructures not accessing memory correctly?
double& DataStructure3D::operator()
  ( unsigned int i, unsigned int j, unsigned int k )
{
  return Data[(i * Size2 + j) * Size3 + k];
}

double DataStructure3D::operator()
  ( unsigned int i, unsigned int j, unsigned int k ) const
{                                                
  return Data[(i * Size2 + j) * Size3 + k];      
}


double DataStructure3D::CellAverage( unsigned int iCF, unsigned int iX, unsigned int nNodes, 
  std::vector<double> Weights )
{

  double avg = 0.0;

  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    avg += Weights[iN] * Data[(iCF * Size2 + iX) * Size3 + iN];
  }

  return avg;
}

// Copy Grid contents into new array
// TODO: Fix DataStructure Copy routines 
// (look at grid -- don't include Guard cells)
void DataStructure3D::copy( std::vector<double> dest )
{

  for ( unsigned int i = 0; i < mSize; i++ )
  {
    dest[i] = Data[i];
  }
}

// zero out structure
void DataStructure3D::zero( )
{
  for ( unsigned int i = 0; i <= mSize; i++ )
  {
    Data[i] = 0.0;
  }
}