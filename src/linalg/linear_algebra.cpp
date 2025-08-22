/**
 * @file linear_algebra.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Basic linear algebra functions.
 *
 * @details Linear algebra routines for quadrature and limiters.
 *          - tri_sym_diag
 *          - invert_matrix
 */

#include <cstddef>
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
void tri_sym_diag(int n, std::vector<double>& d, std::vector<double>& e,
                  std::vector<double>& array) {

  // Parameters for LaPack
  lapack_int m = 0, ldz = 0, info = 0, work_dim = 0;
  m = n;
  char const job = 'V';
  ldz = n;

  if (n == 1) {
    work_dim = 1;
  } else {
    work_dim = 2 * n - 2;
  }

  std::vector<double> ev(n * n);
  std::vector<double> work(work_dim);

  info = LAPACKE_dstev(LAPACK_COL_MAJOR, job, m, d.data(), e.data(), ev.data(),
                       ldz);

  if (info != 0) {
    THROW_ATHELAS_ERROR(
        " ! Issue occurred in initializing quadrature in tri_sym_diag.");
  }

  // Matrix multiply ev' * array. Only Array[0] is nonzero.
  double const k = array[0];
  for (int i = 0; i < n; i++) {
    array[i] = k * ev[static_cast<ptrdiff_t>(n * i)];
  }

  // Update the input vectors with the results
}

/**
 * @brief Use LAPACKE to invert a matrix M using LU factorization.
 **/
void invert_matrix(std::vector<double>& M, int n) {
  lapack_int info1 = 0, info2 = 0;

  std::vector<int> IPIV(n);

  info1 = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, M.data(), n, IPIV.data());
  info2 = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, M.data(), n, IPIV.data());

  if (info1 != 0 || info2 != 0) {
    THROW_ATHELAS_ERROR(" ! Issue occurred in matrix inversion.");
  }
}
