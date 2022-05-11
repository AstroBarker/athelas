#ifndef INITIALIZATION_H
#define INITIALIZATION_H

void InitializeFields( Kokkos::View<double***> uCF, Kokkos::View<double***> uPF,
                       GridStructure& Grid, const unsigned int pOrder,
                       const double GAMMA_IDEAL,
                       const std::string ProblemName );
#endif
