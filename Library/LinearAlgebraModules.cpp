/**
 * File     :  LinearAlgebraModules.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Linear algebra routines needed for quadrature,
 *  Calls LAPACK routine dstev
 **/

#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Error.h"
#include "LinearAlgebraModules.h"
#include "lapacke.h"

// Fill identity matrix
void IdentityMatrix( Kokkos::View<double**> Mat, unsigned int n )
{
  for ( unsigned int i = 0; i < n; i++ )
    for ( unsigned int j = 0; j < n; j++ )
    {
      if ( i == j )
      {
        Mat( i, j ) = 1.0;
      }
      else
      {
        Mat( i, j ) = 0.0;
      }
    }
}

/**
 * Use LAPACKE to diagonalize symmetric tridiagonal matrix with DSTEV
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
  lapack_int m, ldz, info, work_dim;
  m        = n;
  char job = 'V';
  ldz      = n;

  if ( n == 1 )
  {
    work_dim = 1;
  }
  else
  {
    work_dim = 2 * n - 2;
  }

  double* ev   = new double[n * n];
  double* work = new double[work_dim];

  info = LAPACKE_dstev( LAPACK_COL_MAJOR, job, m, d, e, ev, ldz );

  if ( info != 0 )
  {
    throw Error( "Issue occured in initializing quadrature in Tri_Sym_Diag." );
  }

  // Matrix multiply ev' * array. Only Array[0] is nonzero.
  double k = array[0];
  for ( int i = 0; i < n; i++ )
  {
    array[i] = k * ev[n * i];
  }

  delete[] work;
  delete[] ev;
}

/**
 * Use LAPACKE to invert a matrix M using LU factorization.
 **/
void InvertMatrix( double* M, unsigned int n )
{
  lapack_int info1, info2;

  int* IPIV = new int[n];

  info1 = LAPACKE_dgetrf( LAPACK_COL_MAJOR, n, n, M, n, IPIV );
  info2 = LAPACKE_dgetri( LAPACK_COL_MAJOR, n, M, n, IPIV );

  delete[] IPIV;

  if ( info1 != 0 || info2 != 0 )
  {
    throw Error( "Issue occured in matrix inversion." );
  }
}

/**
 * Matrix vector multiplication
 *
 * Parameters:
 * -----------
 * see, e.g.,
 * https://www.netlib.org/blas/
 **/
void MatMul( double alpha, Kokkos::View<double[3][3]> A,
             Kokkos::View<double[3]> x, double beta, Kokkos::View<double[3]> y )
{
  // Calculate A*x=y
  for ( int i = 0; i < 3; i++ )
  {
    for ( int j = 0; j < 3; j++ )
    {
      y( i ) += ( A( i, j ) * x( j ) );
    }
  }
}
