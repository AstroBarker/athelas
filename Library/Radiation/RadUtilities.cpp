/**
 * File     :  RadUtilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Utility routines for radiation fields. Includes Riemann solvers.
 **/

#include <iostream>
#include <vector>
#include <cstdlib>   /* abs */
#include <algorithm> // std::min, std::max

#include "Constants.hpp"
#include "Error.hpp"
#include "Grid.hpp"
#include "PolynomialBasis.hpp"
#include "EquationOfStateLibrary_IDEAL.hpp"
#include "RadUtilities.hpp"

Real FluxRad( Real E, Real F, Real P, Real V, UInt iRF ) {
  assert ( iRF == 1 || iRF == 2 )

  if ( iRF == 0 ) {
    return F - E * V;
  } else {
    return P - F * v;
  }
}
Real SourceRad( Real D, Real V, Real T, Real X, 
                Real E, Real F, Real P, UInt iRF ) {
  assert ( iRF == 1 || iRF == 2 )

  Real a = constants::a;
  Real c = constants::c_cgs;

  Real b = V / c;
  Real term1 = E - a * T*T*T*T - 2.0 * b * F;
  Real term2 = F - E * b - b * P;

  if ( iRF == 0 ) {
    return D * X * term1 + D * X * b * term2;
  } else {
    return D * X * term1 * b D * X * term2;
  }
}
