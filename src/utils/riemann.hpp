#pragma once
/**
 * @file riemann.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Riemann solvers
 */

#include "abstractions.hpp"

namespace riemann {
auto hll( double u_l, double u_r, double f_l, double f_r, double s_l_m, double s_r_p )
    -> double;
} // namespace riemann
