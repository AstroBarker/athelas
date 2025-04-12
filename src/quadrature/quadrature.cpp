/**
 * @file quadrature.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Quadrature rules
 * 
 * @details Computes Gauss-Legendre nodes and weights
 */

#include <iostream>
#include <math.h>

#include "constants.hpp"
#include "linear_algebra.hpp"
#include "quadrature.hpp"

namespace quadrature {

  /**
 * @brief Computes the Jacobi matrix for Legendre-Gauss quadrature rule
 * 
 * @details Constructs the symmetric tridiagonal Jacobi matrix 
 *          needed for Legendre-Gauss quadrature. 
 * 
 * @param m Number of quadrature nodes (matrix dimension)
 * @param aj Output array for matrix diagonal elements
 * @param bj Output array for matrix subdiagonal elements
 * @return Real The zero-th moment (zemu) needed for weight computation
 */
Real Jacobi_Matrix( int m, Real *aj, Real *bj ) {

  Real ab;
  Real zemu;
  Real abi;
  Real abj;

  ab   = 0.0;
  zemu = 2.0 / ( ab + 1.0 );
  for ( int i = 0; i < m; i++ ) {
    aj[i] = 0.0;
  }

  for ( int i = 1; i <= m; i++ ) {
    abi       = i + ab * ( i % 2 );
    abj       = 2 * i + ab;
    bj[i - 1] = sqrt( abi * abi / ( abj * abj - 1.0 ) );
  }

  return zemu;
}

/**
 * @brief Generates a Legendre-Gauss quadrature rule with specified number of 
 *        points
 * 
 * @details This function computes a complete Legendre-Gauss quadrature rule 
 *          with m points. It generates both the quadrature nodes (abscissas) 
 *          and weights for accurate integration of polynomial functions.
 * 
 *          The resulting quadrature rule is optimal for integrating 
 *          polynomial functions up to degree 2m-1.
 * 
 * @param m Number of quadrature points (must be positive)
 * @param nodes Output array for quadrature nodes (abscissas)
 * @param weights Output array for quadrature weights
 */
void LG_Quadrature( int m, Real *nodes, Real *weights ) {
  Real *aj = new Real[m];
  Real *bj = new Real[m];

  Real zemu;

  //  Get the Jacobi matrix and zero-th moment.
  zemu = Jacobi_Matrix( m, aj, bj );

  // Nodes and Weights
  for ( int i = 0; i < m; i++ ) {
    nodes[i] = aj[i];
  }

  weights[0] = sqrt( zemu );
  for ( int i = 1; i < m; i++ ) {
    weights[i] = 0.0;
  }

  // --- Diagonalize the Jacobi matrix. ---
  Tri_Sym_Diag( m, nodes, bj, weights ); // imtqlx

  for ( int i = 0; i < m; i++ ) {
    weights[i] = weights[i] * weights[i];

    // Shift to interval [-0.5, 0.5]
    weights[i] *= 0.5;
    nodes[i] *= 0.5;
  }

  delete[] aj;
  delete[] bj;
}

} // namespace quadrature
