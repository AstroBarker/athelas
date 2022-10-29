#ifndef SLOPELIMITER_UTILITIES_H
#define SLOPELIMITER_UTILITIES_H

#include "Abstractions.hpp"

Real minmod( Real a, Real b, Real c );
Real minmodB( Real a, Real b, Real c, Real dx, Real M );
Real BarthJespersen( Real U_v_L, Real U_v_R, Real U_c_L, Real U_c_T, Real U_c_R,
                     Real alpha );
#endif
