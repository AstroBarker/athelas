/**
 * @file linear_algebra.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Basic linear algebra functions.
 *
 * @details Linear algebra routines for quadrature and limiters.
 *          - Tri_Sym_Diag
 *          - InvertMatrix
 */

#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>

#include "error.hpp"
#include "lapacke.h"
#include "linear_algebra.hpp"

/**
 * @brief Diagonalizes a symmetric tridiagonal matrix using LAPACKE's DSTEV
 * routine
 *
 * @details This function solves the symmetric tridiagonal eigenvalue problem
 *          using LAPACKE's DSTEV routine. It computes both eigenvalues and
 *          eigenvectors, then performs a matrix multiplication with the input
 *          array.
 *
 * @param n The dimension of the matrix
 * @param d Array containing the diagonal elements of the matrix
 * @param e Array containing the subdiagonal elements (length n)
 * @param array Input/output array for the matrix multiplication (Q*)z
 *
 * @throws std::runtime_error if LAPACKE_dstev fails
 *
 * @note This function is used in quadrature initialization.
 */
void Tri_Sym_Diag( int n, Real *d, Real *e, Real *array ) {

  // Parameters for LaPack
  lapack_int m, ldz, info, work_dim;
  m        = n;
  char job = 'V';
  ldz      = n;

  if ( n == 1 ) {
    work_dim = 1;
  } else {
    work_dim = 2 * n - 2;
  }

  Real *ev   = new Real[n * n];
  Real *work = new Real[work_dim];

  info = LAPACKE_dstev( LAPACK_COL_MAJOR, job, m, d, e, ev, ldz );

  if ( info != 0 ) {
    THROW_ATHELAS_ERROR(
        " ! Issue occured in initializing quadrature in Tri_Sym_Diag." );
  }

  // Matrix multiply ev' * array. Only Array[0] is nonzero.
  Real k = array[0];
  for ( int i = 0; i < n; i++ ) {
    array[i] = k * ev[n * i];
  }

  delete[] work;
  delete[] ev;
}

/**
 * @brief Use LAPACKE to invert a matrix M using LU factorization.
 **/
void InvertMatrix( Real *M, int n ) {
  lapack_int info1, info2;

  int *IPIV = new int[n];

  info1 = LAPACKE_dgetrf( LAPACK_COL_MAJOR, n, n, M, n, IPIV );
  info2 = LAPACKE_dgetri( LAPACK_COL_MAJOR, n, M, n, IPIV );

  delete[] IPIV;

  if ( info1 != 0 || info2 != 0 ) {
    THROW_ATHELAS_ERROR( " ! Issue occured in matrix inversion." );
  }
}
