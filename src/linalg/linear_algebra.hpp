#pragma once
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
#include <vector>

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
void tri_sym_diag( int n, std::vector<Real>& d, std::vector<Real>& e,
                   std::vector<Real>& array );
void invert_matrix( std::vector<Real>& M, int n );

/**
 * @brief Testing function: checks A A^-1 = I
 **/
template <typename T>
bool multiply_and_check_identity(T A, T B, double tol = 1e-8) {
    using Scalar = typename T::value_type;
    static_assert(T::rank == 2, "Input views must be rank 2.");

    int N = A.extent(0);
    int K = A.extent(1);
    int M = B.extent(1);

    if (K != B.extent(0) || N != M) {
        std::cerr << "Matrix dimensions are incompatible or result is not square.\n";
        return false;
    }

    // Allocate result matrix
    Kokkos::View<Scalar**, Kokkos::LayoutRight> C("C", N, M);

    // Matrix multiplication: C(i,j) = sum_k A(i,k) * B(k,j)
    Kokkos::parallel_for("MatrixMultiply", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, M}),
        KOKKOS_LAMBDA(int i, int j) {
            Scalar sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        });

    // Check if result is an identity matrix
    auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            Scalar expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(C_host(i, j) - expected) > tol) {
                std::cout << "Mismatch at (" << i << ", " << j << "): " << C_host(i, j)
                          << " != " << expected << "\n";
                return false;
            }
        }
    }

    return true;
}
