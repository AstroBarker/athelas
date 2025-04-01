#ifndef DRIVER_HPP_
#define DRIVER_HPP_

#include "abstractions.hpp"
#include "eos.hpp"
#include "grid.hpp"

int NumNodes( const int order );

Real ComputeCFL( const Real CFL, const int order, const int nStages,
                 const int tOrder );
Real compute_timestep( const View3D<Real> U, const GridStructure *Grid,
                       EOS *eos, const Real CFL, const Options *opts );
#endif // DRIVER_HPP_
