/**
 * Classes for holding multidimensional data
 **/

#include <iostream>
#include "DataStructures.h"

class DataStructure2D
{
public:
  DataStructure2D(unsigned int rows, unsigned int cols);

  // copy constructor
  DataStructure2D(const DataStructure2D &old_obj)
  {
  Rows = old_obj.Rows;
  Cols = old_obj.Cols;

  Data = old_obj.Data;
  }

  double& operator()(unsigned int i, unsigned int j);
  double operator()(unsigned int i, unsigned int j) const;

  ~DataStructure2D()
  {
    delete [] Data;
  }

private:
  unsigned int Rows;
  unsigned int Cols;
  double* Data;
};

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


// This will be e.g., conserved variables
class DataStructure3D
{

public:

  // default constructor - for move constructur
  DataStructure3D()
  {
    Size1 = 1;
    Size2 = 1;
    Size3 = 1;

    mSize = Size1 * Size2 * Size3;

    Data = new double[mSize];
  }
  DataStructure3D( unsigned int N1, unsigned int N2, unsigned int N3 );

  // copy-constructor
  //https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
  DataStructure3D(const DataStructure3D& other)
      : mSize(other.mSize),
        Data(mSize ? new double[mSize] : nullptr)
  {
      // note that this is non-throwing, because of the data
      // types being used; more attention to detail with regards
      // to exceptions must be given in a more general case, however
      std::copy(other.Data, other.Data + mSize, Data);
  }

  friend void swap(DataStructure3D& first, DataStructure3D& second) // nothrow
    {
        // by swapping the members of two objects,
        // the two objects are effectively swapped
        std::swap( first.mSize, second.mSize );
        std::swap( first.Data, second.Data );
    }

  // assignment
  DataStructure3D& operator=(DataStructure3D other) 
  {
      swap(*this, other);

      return *this;
  }

  // move constructor
  DataStructure3D(DataStructure3D&& other) noexcept
      : DataStructure3D() // initialize via default constructor, C++11 only
  {
      swap(*this, other);
  }

  double& operator()( unsigned int i, unsigned int j, unsigned int k );
  double operator()( unsigned int i, unsigned int j, unsigned int k ) const;

  ~DataStructure3D()
  {
    delete [] Data;
  }

private:
  unsigned int Size1;
  unsigned int Size2;
  unsigned int Size3;
  unsigned int mSize;

  double* Data;

};

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