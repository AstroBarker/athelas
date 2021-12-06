/**
 * File     :  SlopeLimiter_Utilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Hold utility functions for the slope limiter class
 *  to keep the class minimal.
 *  
 * Contains:
 * ---------
 * sgn, minmod, minmodB
**/ 

#include <iostream>

#include "SlopeLimiter_Utilities.h"
#include "DataStructures.h"
#include <algorithm>    // std::min, std::max
#include <cstdlib>     /* abs */

// Implements a typesafe sgn function
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
// TODO: make sure this actually has the correct behavior for minmod...
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// Standard minmod function
double minmod( double a, double b, double c )
{
  if ( sgn(a) == sgn(b) && sgn(b) == sgn(c) )
  {
    return sgn(a) * std::min( std::min( a, b ), c );
  }
  else
  {
    return 0.0;
  }
}


// TVB minmod function
double minmodB( double a, double b, double c, double dx, double M )
{
  if ( std::abs( a ) < M * dx * dx )
  {
    return a;
  }
  else
  {
    return minmod( a, b, c );
  }
}