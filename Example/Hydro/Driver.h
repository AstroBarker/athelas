#ifndef DRIVER_H
#define DRIVER_H

typedef Kokkos::View<double*>    DataStructure1DType;
typedef Kokkos::View<double**>   DataStructure2DType;
typedef Kokkos::View<double***>  DataStructure3DType;

int NumNodes( unsigned int order );

double CellAverage( DataStructure3D& U, GridStructure& Grid, ModalBasis& Basis,
                    unsigned int iCF, unsigned int iX );

double ComputeCFL( double CFL, unsigned int order, unsigned int nStages,
                   unsigned int tOrder );
#endif