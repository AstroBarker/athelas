/**
 * @file riemann.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Riemann solvers
 */

#include "abstractions.hpp"

namespace riemann {
auto hll( const double u_l, const double u_r, const double f_l, const double f_r,
          const double s_l_m, const double s_r_p ) -> double {
  constexpr static double eps = 1.0; // TODO(astrobarker) need?
  return ( s_r_p * f_l - s_l_m * f_r + eps * s_r_p * s_l_m * ( u_r - u_l ) ) /
         ( s_r_p - s_l_m );
}
} // namespace riemann
