/**
 * @file riemann.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Riemann solvers
 */

#pragma once

#include "Kokkos_Macros.hpp"
#include <algorithm>

namespace athelas::riemann {
KOKKOS_INLINE_FUNCTION
auto hll(const double u_l, const double u_r, const double f_l, const double f_r,
         const double s_l, const double s_r, const double tau = 1.0) -> double {
  const double eps = std::min(1.0, 1.0 / tau); // TODO(astrobarker) need?
  return (s_r * f_l - s_l * f_r + eps * s_r * s_l * (u_r - u_l)) / (s_r - s_l);
}
} // namespace athelas::riemann
