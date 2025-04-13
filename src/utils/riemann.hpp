#ifndef RIEMANN_HPP_
#define RIEMANN_HPP_
/**
 * @file riemann.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Riemann solvers
 */

#include "abstractions.hpp"

namespace riemann {
auto hll( Real u_l, Real u_r, Real f_l, Real f_r, Real s_l_m, Real s_r_p )
    -> Real;
} // namespace riemann

#endif // RIEMANN_HPP_
