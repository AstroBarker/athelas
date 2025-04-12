/**
 * @file riemann.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Riemann solvers
 */

#include "abstractions.hpp"

Real hll( const Real u_l, const Real u_r, const Real f_l, const Real f_r,
          const Real s_l_m, const Real s_r_p ) {
  return ( s_r_p * f_l - s_l_m * f_r + s_r_p * s_l_m * ( u_r - u_l ) ) /
         ( s_r_p - s_l_m );
}
