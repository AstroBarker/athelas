#pragma once
/**
 * @file utilities.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Useful utilities
 *
 * @details Provides
 *          - SGN
 *          - compute_internal_energy
 *          - to_lower
 */

#include <algorithm>
#include <cctype>
#include <string>

#include "abstractions.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"

namespace utilities {

// [[x]]_+ = -.5 * (x + |x|) is positive part of x
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto pos_part( const double x ) noexcept
    -> double {
  return 0.5 * ( x + std::abs( x ) );
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto EPS( ) {
  return 10 * std::numeric_limits<T>::epsilon( );
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto LARGE( ) {
  return 0.1 * std::numeric_limits<T>::max( );
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto SMALL( ) {
  return 10 * std::numeric_limits<T>::min( );
}

KOKKOS_FORCEINLINE_FUNCTION
auto make_bounded( const double val, const double vmin, const double vmax ) -> double {
  return std::min( std::max( val, vmin + EPS( ) ), vmax * ( 1.0 - EPS( ) ) );
}

// Implements a typesafe SGN function
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-SGN-in-c-c
template <typename T>
KOKKOS_INLINE_FUNCTION
constexpr auto SGN( T val ) -> int {
  return ( T( 0 ) < val ) - ( val < T( 0 ) );
}

template <typename A, typename B>
KOKKOS_INLINE_FUNCTION auto ratio(const A &a, const B &b) {
  const B sgn = b >= 0 ? 1 : -1;
  return a / (b + sgn * SMALL<B>());
}

// nodal specific internal energy
template <class T>
KOKKOS_INLINE_FUNCTION
auto compute_internal_energy( T U, const ModalBasis* basis, const int iX,
                              const int iN ) -> double {
  const double Vel = basis->basis_eval( U, iX, 1, iN );
  const double EmT = basis->basis_eval( U, iX, 2, iN );

  return EmT - ( 0.5 * Vel * Vel );
}

// cell average specific internal energy
template <class T>
KOKKOS_INLINE_FUNCTION
auto compute_internal_energy( T U, const int iX ) -> double {
  return U( 2, iX, 0 ) - ( 0.5 * U( 1, iX, 0 ) * U( 1, iX, 0 ) );
}

// string to_lower function
// adapted from
// http://notfaq.wordpress.com/2007/08/04/cc-convert-string-to-upperlower-case/
template <class T>
auto to_lower( T data ) -> T {
  std::transform( data.begin( ), data.end( ), data.begin( ),
                  []( unsigned char c ) { return std::tolower( c ); } );
  return data;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
auto l2_norm(const ViewType& v) -> double {
    static_assert(ViewType::rank == 1, "l2_norm expects rank-1 views.");
    const int N = v.extent(0);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += v(i) * v(i);
    }
    return std::sqrt(sum) / N;
}
template <typename ViewType>
KOKKOS_INLINE_FUNCTION
auto l2_norm_diff(const ViewType& a, const ViewType& b) -> double {
    static_assert(ViewType::rank == 1, "l2_norm_diff expects rank-1 views.");
    const int N = a.extent(0);
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        sum += (a(k) - b(k)) * (a(k) - b(k));
    }
    return std::sqrt(sum) / N;
}
template <typename ViewType>
KOKKOS_INLINE_FUNCTION
auto l2_norm(const ViewType& v, int iC) -> double {
    static_assert(ViewType::rank == 2, "l2_norm expects rank-2 views.");
    const int N = v.extent(1);
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        sum += v(iC, k) * v(iC, k);
    }
    return std::sqrt(sum) / N;
}
template <typename ViewType1, typename ViewType2>
KOKKOS_INLINE_FUNCTION
auto l2_norm_diff(const ViewType1& a, const ViewType2& b, int iC) -> double {
    static_assert(ViewType1::rank == 2, "l2_norm_diff expects rank-2 views.");
    static_assert(ViewType2::rank == 2, "l2_norm_diff expects rank-2 views.");
    const int N = a.extent(1);
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        sum += (a(iC, k) - b(iC, k)) * (a(iC, k) - b(iC, k));
    }
    return std::sqrt(sum) / N;
}

} // namespace utilities
