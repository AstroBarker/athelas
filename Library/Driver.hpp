#ifndef _DRIVER_HPP_
#define _DRIVER_HPP_

#include "Abstractions.hpp"

int NumNodes( unsigned int order );

Real ComputeCFL( Real CFL, unsigned int order, unsigned int nStages,
                 unsigned int tOrder );
#endif // _DRIVER_HPP_
