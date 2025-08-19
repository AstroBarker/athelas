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

#include "grid.hpp"
#include "polynomial_basis.hpp"
#include "problem_in.hpp"
#include "slope_limiter.hpp"
#include "state.hpp"

struct GridType {
  double r{};
};

struct DataType {
  double x{};
};

void write_state(State* state, GridStructure& grid, SlopeLimiter* SL,
                 ProblemIn* pin, double time, int order, int i_write,
                 bool do_rad);

void print_simulation_parameters(GridStructure grid, ProblemIn* pin);

void write_basis(ModalBasis* basis, int ihi, int nNodes, int order,
                 const std::string& problem_name);
