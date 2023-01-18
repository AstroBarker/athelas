#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <math.h> /* atan */

#include "Abstractions.hpp"


constexpr Real PI( ) { return std::atan( 1 ) * 4; }
constexpr Real G_GRAV = 6.674299999999999e-8; // cgs
constexpr Real L_sun  = 3.828e33;             // cgs
constexpr Real a = 7.5657e-15; //cgs
constexpr Real N_A = 6.022e+23;
constexpr Real k_B = 1.3806e-16;
#endif
