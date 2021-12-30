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
#include <limits>

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


/** 
 *  Barth-Jespersen limiter
 *  Parameters:
 *  -----------
 *  U_vertex: array of vertex values on given cell
 *  U_c: array of cell averages of current cell + neighbors 
 *    [ left, current, right ]
 *  alpha: scaling coefficient for BJ limiter. 
 *    alpha=1 is classical limiter, alpha=0 enforces constant solutions
**/
double BarthJespersen( double* U_vertex, double* U_c, double alpha )
{
  // Get U_min, U_max
  double U_min = 10000000.0 * U_c[1];
  double U_max = std::numeric_limits<double>::epsilon() * U_c[1];

  for( unsigned int i = 0; i < 3; i++ )
  {
    U_min = std::min( U_min, U_c[i] );
    U_max = std::max( U_max, U_c[i] );
  }

  // loop of cell certices
  double phi = 10000000000.0;
  double phi_L = 0.0;
  double phi_R = 0.0;

  // left vertex
  if ( U_vertex[0] - U_c[1] > 0.0 )
  {
    phi_L = std::min( 1.0, alpha * (U_max - U_c[1]) / (U_vertex[0] - U_c[1]) ); 
  }
  else if ( U_vertex[0] - U_c[1] == 0.0 )
  {
    phi_L = 1.0;
  }
  else
  {  
    phi_L = std::min( 1.0, alpha * (U_min - U_c[1]) / (U_vertex[0] - U_c[1]) ); 
  }

  // right vertex
  if ( U_vertex[1] - U_c[1] > 0.0 )
  {
    phi_R = std::min( 1.0, alpha * (U_max - U_c[1]) / (U_vertex[1] - U_c[1]) ); 
  }
  else if ( U_vertex[1] - U_c[1] == 0.0 )
  {
    phi_R = 1.0;
  }
  else
  {  
    phi_R = std::min( 1.0, alpha * (U_min - U_c[1]) / (U_vertex[1] - U_c[1]) ); 
  }

  // return min of two values
  return std::min( phi_L, phi_R );
}
