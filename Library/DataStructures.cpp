/**
 * Classes for holding multidimensional data
 **/

#include "DataStructures.h"

DataStructure2D::DataStructure2D(unsigned int rows, unsigned int cols)
{
  Rows = rows;
  Cols = cols;

  mSize = rows * cols;

  Data = new double[rows * cols];
}

//TODO: When we use this, make sure it is accessing data efficiently.
double& DataStructure2D::operator()(unsigned int i, unsigned int j)
{
  return Data[i * Cols + j];
}

double DataStructure2D::operator()(unsigned int i, unsigned int j) const
{
  return Data[i * Cols + j];
}

// init as {nCF, nX, nNodes}
DataStructure3D::DataStructure3D( unsigned int N1, unsigned int N2, unsigned int N3 )
{
  Size1 = N1; // nCF
  Size2 = N2; // nX
  Size3 = N3; // nNodes

  mSize = Size1 * Size2 * Size3;

  Data = new double[mSize];
}

// access (iCF, iX, iN)
double& DataStructure3D::operator()
  ( unsigned int i, unsigned int j, unsigned int k )
{
  return Data[k + j*Size3 + i * Size3 * Size2];
}

double DataStructure3D::operator()
  ( unsigned int i, unsigned int j, unsigned int k ) const
{                                                
  return Data[k + j*Size3 + i * Size3 * Size2];      
}

// Copy Grid contents into new array
// TODO: Fix DataStructure Copy routines 
// (look at grid -- don't include Guard cells)
void DataStructure3D::copy( double* dest )
{

  for ( unsigned int i = 0; i < mSize; i++ )
  {
    dest[i] = Data[i];
  }
}