#ifndef UTILITIES_H
#define UTILITIES_H

// Implements a typesafe sgn function
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T>
int sgn( T val )
{
  return ( T( 0 ) < val ) - ( val < T( 0 ) );
}

#endif