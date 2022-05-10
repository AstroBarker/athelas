#ifndef IOLIBRARY_H
#define IOLIBRARY_H

#include "Kokkos_Core.hpp"

struct GridType
{
  double r{ };
};

struct DataType
{
  double x{ };
};

void WriteState( Kokkos::View<double***> uCF, Kokkos::View<double***> uPF,
                 Kokkos::View<double***> uAF, GridStructure& Grid,
                 SlopeLimiter& SL, const std::string ProblemName );

void PrintSimulationParameters( GridStructure& Grid, unsigned int pOrder,
                                unsigned int tOrder, unsigned int nStages,
                                double CFL, double alpha, double TCI,
                                bool Char_option, bool TCI_Option,
                                std::string ProblemName );

void WriteBasis( ModalBasis& Basis, unsigned int ilo, unsigned int ihi,
                 unsigned int nNodes, unsigned int order,
                 std::string ProblemName );

#endif