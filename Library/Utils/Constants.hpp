#ifndef _CONSTANTS_HPP_
#define _CONSTANTS_HPP_

#include <math.h> /* atan */

#include "Abstractions.hpp"

namespace constants {

constexpr Real PI( ) { return std::atan( 1 ) * 4; }
constexpr Real G_GRAV = 6.674299999999999e-8; // cgs
constexpr Real L_sun  = 3.828e33;             // cgs
constexpr Real M_sun  = 1.98840987e+33;       // cgs
constexpr Real sigma_sb = 5.670374419184e-5;  // cgs
constexpr Real a = 7.5657e-15;                // cgs
constexpr Real k_B  = 1.380649e-16;           // cgs
constexpr Real m_e  = 9.1093837e-28;          // cgs
constexpr Real h    = 6.62607015e-27;         // cgs
constexpr Real hbar = 1.05457182e-27;         // cgs
constexpr Real N_A  = 6.02214076e+23;         // 1 / mol

} // namespace constants
#endif // _CONSTANTS_HPP_
