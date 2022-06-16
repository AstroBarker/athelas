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
#include <algorithm> // std::min, std::max
#include <cstdlib>   /* abs */

#include "Utilities.h"
#include "DataStructures.h"
#include "SlopeLimiter_Utilities.h"

// Standard minmod function
double minmod( double a, double b, double c )
{
  if ( sgn( a ) == sgn( b ) && sgn( b ) == sgn( c ) )
  {
    return sgn( a ) * std::min( std::min( a, b ), c );
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
 *  U_v_*: left/right vertex values on target cell
 *  U_c_*: cell averages of target cell + neighbors
 *    [ left, target, right ]
 *  alpha: scaling coefficient for BJ limiter.
 *    alpha=1 is classical limiter, alpha=0 enforces constant solutions
 **/
double BarthJespersen( double U_v_L, double U_v_R, double U_c_L, double U_c_T,
                       double U_c_R, double alpha )
{
  // Get U_min, U_max
  double U_min_L = 10000000.0 * U_c_T;
  double U_min_R = 10000000.0 * U_c_T;
  double U_max_L = std::numeric_limits<double>::epsilon( ) * U_c_T * 0.00001;
  double U_max_R = std::numeric_limits<double>::epsilon( ) * U_c_T * 0.00001;

  U_min_L = std::min( U_min_L, std::min( U_c_T, U_c_L ) );
  U_max_L = std::max( U_max_L, std::max( U_c_T, U_c_L ) );
  U_min_R = std::min( U_min_R, std::min( U_c_T, U_c_R ) );
  U_max_R = std::max( U_max_R, std::max( U_c_T, U_c_R ) );

  // loop of cell certices
  double phi_L = 0.0;
  double phi_R = 0.0;

  // left vertex
  if ( U_v_L - U_c_T + 1.0 > 1.0 )
  {
    phi_L = std::min( 1.0, alpha * ( U_max_L - U_c_T ) / ( U_v_L - U_c_T ) );
  }
  else if ( U_v_L - U_c_T + 1.0 < 1.0 )
  {
    phi_L = std::min( 1.0, alpha * ( U_min_L - U_c_T ) / ( U_v_L - U_c_T ) );
  }
  else
  {
    phi_L = 1.0;
  }

  // right vertex
  if ( U_v_R - U_c_T + 1.0 > 1.0 )
  {
    phi_R = std::min( 1.0, alpha * ( U_max_R - U_c_T ) / ( U_v_R - U_c_T ) );
  }
  else if ( U_v_R - U_c_T + 1.0 < 1.0 )
  {
    phi_R = std::min( 1.0, alpha * ( U_min_R - U_c_T ) / ( U_v_R - U_c_T ) );
  }
  else
  {
    phi_R = 1.0;
  }

  // return min of two values
  return std::min( phi_L, phi_R );
}
