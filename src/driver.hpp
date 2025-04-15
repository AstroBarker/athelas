#ifndef DRIVER_HPP_
#define DRIVER_HPP_
/**
 * @file driver.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Driver
 *
 * @details Functions:
 *            - NumNodes
 *            - compute_cfl
 *            - compute_timestep
 */

#include "abstractions.hpp"
#include "eos.hpp"
#include "grid.hpp"

// auto compute_cfl( Real CFL, int order, int nStages, int tOrder ) -> Real;
// auto compute_timestep( View3D<Real> U, const GridStructure* Grid,
//                        EOS* eos, Real CFL, const Options* opts ) -> Real;
#endif // DRIVER_HPP_
