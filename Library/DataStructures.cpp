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