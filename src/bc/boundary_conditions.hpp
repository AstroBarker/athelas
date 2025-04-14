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

namespace bc {
void apply_bc( View3D<Real> uCF, const GridStructure* Grid, int order,
               const std::string& BC );
} // namespace bc
#endif // BOUNDARY_CONDITIONS_HPP_
