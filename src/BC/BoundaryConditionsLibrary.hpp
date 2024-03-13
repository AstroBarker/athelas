#ifndef BOUNDARYCONDITIONSLIBRARY_HPP_
#define BOUNDARYCONDITIONSLIBRARY_HPP_

#include "Abstractions.hpp"
#include "Grid.hpp"

void ApplyBC( View3D<Real> uCF, GridStructure *Grid, const int order,
              const std::string BC );

#endif // BOUNDARYCONDITIONSLIBRARY_HPP_
