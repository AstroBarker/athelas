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

void WriteState( Kokkos::View<Real ***> uCF, Kokkos::View<Real ***> uPF,
                 Kokkos::View<Real ***> uAF, GridStructure *Grid,
                 SlopeLimiter *SL, const std::string ProblemName, Real time,
                 UInt order, int i_write );

void PrintSimulationParameters( GridStructure *Grid, UInt pOrder, UInt tOrder,
                                UInt nStages, Real CFL, Real alpha, Real TCI,
                                bool Char_option, bool TCI_Option,
                                std::string ProblemName );

void WriteBasis( ModalBasis *Basis, UInt ilo, UInt ihi, UInt nNodes, UInt order,
                 std::string ProblemName );

#endif
