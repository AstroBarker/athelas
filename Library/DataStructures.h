#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

/**
 * File     :  DataStructures.h
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Classes for holding multidimensional data.
 *  Multi-D structures are wrapped around 1D vectors to conveniently and 
 *  efficiently access data. For conserved variables data structure, 
 *  initialize/acces as DataStructures3D uCF(nCF,nX,order)
 * Contains : DataStructures2D, DataStructures3D
**/ 

#include <algorithm>    // std::copy
#include <vector>

class DataStructure2D
{
public:

  DataStructure2D( unsigned int rows, unsigned int cols );

  double& operator()(unsigned int i, unsigned int j);
  double operator()(unsigned int i, unsigned int j) const;

private:
  unsigned int Rows;
  unsigned int Cols;
  unsigned int mSize;
  std::vector<double> Data;
};


// This will be e.g., conserved variables
class DataStructure3D
{

public:

  DataStructure3D( unsigned int N1, unsigned int N2, unsigned int N3 );

  double& operator()( unsigned int i, unsigned int j, unsigned int k );
  double operator()( unsigned int i, unsigned int j, unsigned int k ) const;

  void zero( );

private:
  unsigned int Size1;
  unsigned int Size2;
  unsigned int Size3;
  unsigned int mSize;

  std::vector<double> Data;

};

#endif
