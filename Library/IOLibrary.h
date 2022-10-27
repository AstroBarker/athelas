#ifndef IOLIBRARY_H
#define IOLIBRARY_H

#include "Kokkos_Core.hpp"
#include "Abstractions.hpp"

struct GridType
{
  Real r{ };
};

struct DataType
{
  Real x{ };
};

void WriteState( Kokkos::View<Real***> uCF, Kokkos::View<Real***> uPF,
                 Kokkos::View<Real***> uAF, GridStructure& Grid,
                 SlopeLimiter& SL, const std::string ProblemName, Real time,
                 unsigned int order, int i_write );

void PrintSimulationParameters( GridStructure& Grid, unsigned int pOrder,
                                unsigned int tOrder, unsigned int nStages,
                                Real CFL, Real alpha, Real TCI,
                                bool Char_option, bool TCI_Option,
                                std::string ProblemName );

void WriteBasis( ModalBasis& Basis, unsigned int ilo, unsigned int ihi,
                 unsigned int nNodes, unsigned int order,
                 std::string ProblemName );

#endif
