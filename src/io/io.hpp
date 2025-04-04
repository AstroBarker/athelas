#ifndef IO_HPP_
#define IO_HPP_

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"
#include "problem_in.hpp"
#include "slope_limiter.hpp"
// #include "slope_limiter_weno.hpp"
// #include "slope_limiter_tvdminmod.hpp"
#include "state.hpp"

struct GridType {
  Real r{ };
};

struct DataType {
  Real x{ };
};

void WriteState( State *state, GridStructure Grid, SlopeLimiter *SL,
                 const std::string ProblemName, Real time, int order,
                 int i_write, bool do_rad );

void PrintSimulationParameters( GridStructure Grid, ProblemIn *pin,
                                const Real CFL );

void WriteBasis( ModalBasis *Basis, unsigned int ilo, unsigned int ihi,
                 unsigned int nNodes, unsigned int order,
                 std::string ProblemName );

#endif // IO_HPP_
