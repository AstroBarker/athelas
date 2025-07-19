#pragma once
/**
 * @file constants.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Physical constants
 *
 * Hopefully consistent with astropy
 * TODO(astrobarker): implement different
 * unit systems
 */

#include <numbers>

namespace constants {

constexpr double PI       = std::numbers::pi;
constexpr double G_GRAV   = 6.674299999999999e-8; // cgs
constexpr double L_sun    = 3.828e33; // cgs
constexpr double M_sun    = 1.98840987e+33; // cgs
constexpr double sigma_sb = 5.670374419184431e-5; // cgs
constexpr double a        = 7.5657332502800007e-15; // cgs
constexpr double k_B      = 1.380649e-16; // cgs
constexpr double m_e      = 9.1093837015e-28; // cgs
constexpr double m_p      = 1.67262192369e-24; // cgs
constexpr double h        = 6.62607015e-27; // cgs
constexpr double hbar     = 1.05457182e-27; // cgs
constexpr double N_A      = 6.02214076e+23; // 1 / mol
constexpr double c_cgs    = 2.99792458e+10; // cgs
constexpr double c        = 1.0; // natural

} // namespace constants
