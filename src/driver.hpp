#ifndef DRIVER_HPP_
#define DRIVER_HPP_

#include "abstractions.hpp"

int NumNodes( int order );

Real ComputeCFL( const Real CFL, const int order, const int nStages,
                 const int tOrder );
#endif // DRIVER_HPP_
