#pragma once
/**
 * @file io.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief HDF5 and std out IO routines
 *
 * @details Collection of functions for IO
 */

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

void write_state( State* state, GridStructure grid, SlopeLimiter* SL,
                  const std::string& problem_name, Real time, int order,
                  int i_write, bool do_rad );

void print_simulation_parameters( GridStructure grid, ProblemIn* pin,
                                  Real CFL );

void write_basis( ModalBasis* basis, unsigned int ilo, unsigned int ihi,
                  unsigned int nNodes, unsigned int order,
                  const std::string& problem_name );
