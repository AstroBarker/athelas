#ifndef BOUNDARYCONDITIONSLIBRARY_H
#define BOUNDARYCONDITIONSLIBRARY_H

void ApplyBC_Fluid( Kokkos::View<double***> uCF, GridStructure& Grid,
                    unsigned int order, const std::string BC );

#endif