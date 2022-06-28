#ifndef BOUNDARYCONDITIONSLIBRARY_H
#define BOUNDARYCONDITIONSLIBRARY_H

void ApplyBC_Fluid( Kokkos::View<double***> uCF, const GridStructure& Grid,
                    const unsigned int order, const std::string BC );

#endif