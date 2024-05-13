#ifndef IO_LIBRARY_HPP_
#define IO_LIBRARY_HPP_

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"
#include "problem_in.hpp"
#include "slope_limiter.hpp"
#include "state.hpp"

struct GridType {
  Real r{ };
};

struct DataType {
  Real x{ };
};

void WriteState( State *state, GridStructure Grid, SlopeLimiter *SL,
                 const std::string ProblemName, Real time, int order,
                 int i_write );

void PrintSimulationParameters( GridStructure Grid, ProblemIn *pin,
                                const Real CFL );

void WriteBasis( ModalBasis *Basis, unsigned int ilo, unsigned int ihi,
                 unsigned int nNodes, unsigned int order,
                 std::string ProblemName );

#endif // IO_LIBRARY_HPP_
