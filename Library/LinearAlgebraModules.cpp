/** 
 * Linear algebra routines
 **/

#include <iostream>

#include "LinearAlgebraModules.h"

// extern "C" {
extern "C" int dstev_(char* job, int* N, double* D, double* OFFD, double* EV, int* VDIM, double* WORK, int* INFO);
// extern "C" void dsyev_(char* job, int* N, double* D, double* OFFD, double** EV, int* VDIM, double* WORK, int* INFO);
// }

/**
 * Construct an n by m matrix allocated on the heap.
 * 
 * Parameters:
 * 
 *   int rows
 *   int cols
 **/
double** AllocateMatrix( int rows, int cols )
{
  double** mat;
  mat = new double* [rows];
  for (int i=0; i<rows; i++)
  {
      mat[i] = new double[cols];
  }
  return mat;
}


/**
 * Deallocate array memory.
 */
void DeallocateMatrix( double** A, int rows )
{
  //Deallocate each sub-array
  for(int i = 0; i < rows; ++i) {
      delete[] A[i];   
  }
  //Deallocate the array of pointers
  delete[] A;
}


/**
 * This function multiplies two matrices. 
 * 
 * Parameters:
 * 
 *   double** A, B      : arrays to multiply
 *   int rows_A, cols_A : dimensions of A
 *   int rows_B, cols_B : dimensions of B
 */
double** matmul( double** A, double** B,
    int rows_A, int cols_A, int rows_B, int cols_B )
{
  // Make sure that the dimensions are okay!
  // Otherwise return error_mat() to indicate the error.
  if (cols_A != rows_B)
  {
      return A;
  }

  // Construct a new matrix to hold the product
  double** mult = AllocateMatrix( rows_A, cols_B );
  double c; // holds mult[i][j]

  // #pragma omp parallel for
  for (int i = 0; i < rows_A; i++)
  {
    for (int j = 0; j < cols_B; j++)
    {
      mult[i][j] = 0;
      c = mult[i][j];
      for (int k = 0; k < cols_A; k++)
      {
          c += A[i][k] * B[k][j];
      }
      mult[i][j] = c;
    // printf("%f \n", mult[i][j]);
    }
  }   

  return mult;
}


/**
 * Use LaPack to diagonalize symmetric tridiagonal matrix with DSTEV
 * 
 * Parameters:
 *
 *   n: matrix dimension
 *   d: diagonal array of matrix 
 *   r: subdiagonal array (length n)
 *   array: product (Q*)z (in/output)   
 **/
void Tri_Sym_Diag( int n, double* d, double* e, double* array )
{

  // Parameters for LaPack
  char job = 'V';
  int info;
  int ldz = n;
  int work_dim;

  if ( n == 1 )
  {
    work_dim = 1;
  }else{
    work_dim = 2 * n - 2;
  }

  double* ev = new double[n*n];
  double* work = new double[work_dim];
  
  dstev_(&job, &n, d, e, &*ev, &ldz, work, &info);
  

  // Matrix mu;tiply ev' * array. Only Array[0] is nonzero.
  double k = array[0];
  for ( int i = 0; i < n; i++ )
  {
    array[i] = k * ev[i];
  }

  delete [] work;
  delete [] ev;
}