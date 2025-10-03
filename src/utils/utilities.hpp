#pragma once
/**
 * @file utilities.hpp
 * --------------
 *
 * @brief Useful utilities
 *
 * @details Provides
 *          - LINTERP
 *          - SGN
 *          - compute_internal_energy
 *          - to_lower
 */

#include <algorithm>
#include <cctype>

#include "Kokkos_Macros.hpp"
#include "polynomial_basis.hpp"

namespace athelas::utilities {

/**
 * @brief simple linear interpolation to x
 *
 * Uses fused multiply add (std::fma) to reduce rounding errors when available.
 */
KOKKOS_FUNCTION
template <typename T>
constexpr auto LINTERP(T x0, T x1, T y0, T y1, T x) noexcept -> T {
  if (x0 == x1) {
    return y0;
  }
  const T t = (x - x0) / (x1 - x0);
  return std::fma(y1 - y0, t, y0);
}

// [[x]]_+ = -.5 * (x + |x|) is positive part of x
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto pos_part(const double x) noexcept
    -> double {
  return 0.5 * (x + std::abs(x));
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto EPS() {
  return 10 * std::numeric_limits<T>::epsilon();
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto LARGE() {
  return 0.1 * std::numeric_limits<T>::max();
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto SMALL() {
  return 10 * std::numeric_limits<T>::min();
}

KOKKOS_FORCEINLINE_FUNCTION
auto make_bounded(const double val, const double vmin, const double vmax)
    -> double {
  return std::min(std::max(val, vmin + EPS()), vmax * (1.0 - EPS()));
}

// Implements a typesafe SGN function
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-SGN-in-c-c
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr auto SGN(T val) -> int {
  return (T(0) < val) - (val < T(0));
}

template <typename A, typename B>
KOKKOS_INLINE_FUNCTION auto ratio(const A &a, const B &b) {
  const B sgn = b >= 0 ? 1 : -1;
  return a / (b + sgn * SMALL<B>());
}

// nodal specific internal energy
template <class T>
KOKKOS_INLINE_FUNCTION auto compute_internal_energy(T U,
                                                    const ModalBasis *basis,
                                                    const int ix, const int iN)
    -> double {
  const double Vel = basis->basis_eval(U, ix, 1, iN);
  const double EmT = basis->basis_eval(U, ix, 2, iN);

  return EmT - (0.5 * Vel * Vel);
}

// cell average specific internal energy
template <class T>
KOKKOS_INLINE_FUNCTION auto compute_internal_energy(T U, const int ix)
    -> double {
  return U(ix, 0, 2) - (0.5 * U(ix, 0, 1) * U(ix, 0, 1));
}

// string to_lower function
// adapted from
// http://notfaq.wordpress.com/2007/08/04/cc-convert-string-to-upperlower-case/
template <class T>
KOKKOS_INLINE_FUNCTION auto to_lower(T data) -> T {
  std::transform(data.begin(), data.end(), data.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return data;
}

} // namespace athelas::utilities
