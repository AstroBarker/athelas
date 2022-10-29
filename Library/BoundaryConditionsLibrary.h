#ifndef BOUNDARYCONDITIONSLIBRARY_H
#define BOUNDARYCONDITIONSLIBRARY_H

#include "Abstractions.hpp"

void ApplyBC_Fluid( Kokkos::View<Real***> uCF, GridStructure *Grid,
                    const unsigned int order, const std::string BC );

#endif
