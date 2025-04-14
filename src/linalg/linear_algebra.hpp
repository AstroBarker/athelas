#ifndef _LINEAR_ALGEBRA_HPP_
#define LINEAR_ALGEBRA_HPP_
/**
 * @file linear_algebra.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Basic linear algebra functions.
 *
 * @details Linear algebra routines for quadrature and limiters.
 *          - tri_sym_diag
 *          - invert_matrix
 */

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"

// Fill identity matrix
template <class T>
constexpr void IDENTITY_MATRIX( T Mat, int n ) {
  for ( int i = 0; i < n; i++ ) {
    for ( int j = 0; j < n; j++ ) {
      if ( i == j ) {
        Mat( i, j ) = 1.0;
      } else {
        Mat( i, j ) = 0.0;
      }
    }
  }
}

/**
 * @brief Matrix vector multiplication
 **/
template <class M, class V>
constexpr void MAT_MUL( Real /*alpha*/, M A, V x, Real /*beta*/, V y ) {
  // Calculate A*x=y
  for ( int i = 0; i < 3; i++ ) {
    for ( int j = 0; j < 3; j++ ) {
      y( i ) += ( A( i, j ) * x( j ) );
    }
  }
}
void tri_sym_diag( int n, Real* d, Real* e, Real* array );
void invert_matrix( Real* M, int n );

#endif // _LINEAR_ALGEBRA_HPP_
