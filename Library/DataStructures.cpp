/**
 * Classes for holding multidimensional data
 **/


class DataStructure2D
{
public:
  DataStructure2D(unsigned int rows, unsigned int cols);
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
  DataStructure3D( unsigned int N1, unsigned int N2, unsigned int N3 );
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
  double* Data;

};

// init as {nNodes, nX, nCF}
DataStructure3D::DataStructure3D( unsigned int N1, unsigned int N2, unsigned int N3 )
{
  Size1 = N1;
  Size2 = N2;
  Size3 = N3;

  Data = new double[N1 * N2 * N3];
}

double& DataStructure3D::operator()( unsigned int i, unsigned int j, unsigned int k )
{
  return Data[i + Size2*j + Size2*Size3*k];
}

double DataStructure3D::operator()( unsigned int i, unsigned int j, unsigned int k ) const
{                                                
  return Data[i + Size2*j + Size2*Size3*k];      
}

