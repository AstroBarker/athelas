#ifndef BOUNDARY_CONDITIONS_HPP_
#define BOUNDARY_CONDITIONS_HPP_

/**
 * @file boundary_conditions.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Boundary conditions
 * 
 * @details Implemented BCs
 *            - reflecting
 *            - periodic
 *            - homogenous (default)
 *            - shockless noh
 */

#include "abstractions.hpp"
#include "grid.hpp"

void ApplyBC( View3D<Real> uCF, GridStructure *Grid, const int order,
              const std::string BC );

#endif // BOUNDARY_CONDITIONS_HPP_
