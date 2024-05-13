#ifndef BOUNDARY_CONDITIONS_LIBRARY_HPP_
#define BOUNDARY_CONDITIONS_LIBRARY_HPP_

#include "abstractions.hpp"
#include "grid.hpp"

void ApplyBC( View3D<Real> uCF, GridStructure *Grid, const int order,
              const std::string BC );

#endif // BOUNDARY_CONDITIONS_LIBRARY_HPP_
