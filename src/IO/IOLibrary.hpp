#ifndef _IOLIBRARY_HPP_
#define _IOLIBRARY_HPP_

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Grid.hpp"
#include "PolynomialBasis.hpp"
#include "ProblemIn.hpp"
#include "SlopeLimiter.hpp"
#include "state.hpp"

struct GridType {
  Real r{ };
};

struct DataType {
  Real x{ };
};

void WriteState( State *state, GridStructure Grid, SlopeLimiter *SL,
                 const std::string ProblemName, Real time, UInt order,
                 int i_write );

void PrintSimulationParameters( GridStructure Grid, ProblemIn *pin,
                                const Real CFL );

void WriteBasis( ModalBasis *Basis, UInt ilo, UInt ihi, UInt nNodes, UInt order,
                 std::string ProblemName );

#endif // _IOLIBRARY_HPP_
