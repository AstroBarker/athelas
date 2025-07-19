#pragma once
/**
 * @file riemann.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Riemann solvers
 */

namespace riemann {
auto hll(double u_l, double u_r, double f_l, double f_r, double s_l, double s_r,
         double tau = 1.0) -> double;
} // namespace riemann
