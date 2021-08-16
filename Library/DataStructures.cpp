/**
 * Classes for holding multidimensional data
 **/

#include "DataStructures.h"

DataStructure2D::DataStructure2D(unsigned int rows, unsigned int cols)
{
  Rows = rows;
  Cols = cols;

  Data = new double[rows * cols];
}

double& DataStructure2D::operator()(unsigned int i, unsigned int j)
{
  return Data[i * Cols + j];
}

double DataStructure2D::operator()(unsigned int i, unsigned int j) const
{
  return Data[i * Cols + j];
}

// init as {nNodes, nX, nCF}
DataStructure3D::DataStructure3D( unsigned int N1, unsigned int N2, unsigned int N3 )
{
  Size1 = N1;
  Size2 = N2;
  Size3 = N3;

  mSize = Size1 * Size2 * Size3;

  Data = new double[mSize];
}

double& DataStructure3D::operator()
  ( unsigned int i, unsigned int j, unsigned int k )
{
  return Data[i + Size2*j + Size2*Size3*k];
}

double DataStructure3D::operator()
  ( unsigned int i, unsigned int j, unsigned int k ) const
{                                                
  return Data[i + Size2*j + Size2*Size3*k];      
}

// Copy Grid contents into new array
void DataStructure3D::copy( double* dest )
{

  for ( unsigned int i = 0; i < mSize; i++ )
  {
    dest[i] = Data[i];
  }
}