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

#ifndef UTILITIES_HPP_
#define UTILITIES_HPP_

#include <algorithm>
#include <cctype>
#include <string>

#include "abstractions.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"

namespace utilities {

// [[x]]_+ = -.5 * (x + |x|) is positive part of x
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto pos_part(const Real x) noexcept -> Real {
  return 0.5 * (x + std::abs(x));
}

// Implements a typesafe SGN function
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-SGN-in-c-c
template <typename T>
constexpr auto SGN( T val ) -> int {
  return ( T( 0 ) < val ) - ( val < T( 0 ) );
}

// nodal specific internal energy
template <class T>
auto compute_internal_energy( T U, const ModalBasis* basis, const int iX,
                              const int iN ) -> Real {
  const Real Vel = basis->basis_eval( U, iX, 1, iN );
  const Real EmT = basis->basis_eval( U, iX, 2, iN );

  return EmT - ( 0.5 * Vel * Vel );
}

// cell average specific internal energy
template <class T>
auto compute_internal_energy( T U, const int iX ) -> Real {
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

} // namespace utilities

#endif // UTILITIES_HPP_
