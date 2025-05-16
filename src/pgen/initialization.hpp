#pragma once
/**
 * @file initialization.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Top level problem initialization
 *
 * @details Calls specific problem pgen functions.
 */

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "advection.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"
#include "pgen/moving_contact.hpp"
#include "pgen/noh.hpp"
#include "pgen/problem_in.hpp"
#include "pgen/marshak.hpp"
#include "pgen/rad_advection.hpp"
#include "pgen/rad_equilibrium.hpp"
#include "pgen/rad_shock_steady.hpp"
#include "pgen/rad_shock.hpp"
#include "pgen/sedov.hpp"
#include "pgen/shockless_noh.hpp"
#include "pgen/shu_osher.hpp"
#include "pgen/smooth_flow.hpp"
#include "pgen/sod.hpp"
#include "state.hpp"

/**
 * Initialize the conserved Fields for various problems.
 * TODO: For now I initialize constant on each cell. Is there a better way?
 * TODO: To be good Kokkos, either make all relevant loops par_for,
 * or a a device-host copy
 **/
void initialize_fields( State* state, GridStructure* grid, const EOS* /*eos*/,
                        const ProblemIn* pin ) {

  const std::string problem_name = pin->problem_name;

  if ( problem_name == "sod" ) {
    sod_init( state, grid, pin );
  } else if ( problem_name == "shu_osher" ) {
    shu_osher_init( state, grid, pin );
  } else if ( problem_name == "moving_contact" ) {
    moving_contact_init( state, grid, pin );
  } else if ( problem_name == "smooth_advection" ) {
    advection_init( state, grid, pin );
  } else if ( problem_name == "sedov" ) {
    sedov_init( state, grid, pin );
  } else if ( problem_name == "noh" ) {
    noh_init( state, grid, pin );
  } else if ( problem_name == "shockless_noh" ) {
    shockless_noh_init( state, grid, pin );
  } else if ( problem_name == "smooth_flow" ) {
    smooth_flow_init( state, grid, pin );
  } else if ( problem_name == "rad_equilibrium" ) {
    rad_equilibrium_init( state, grid, pin );
  } else if ( problem_name == "rad_advection" ) {
    rad_advection_init( state, grid, pin );
  } else if ( problem_name == "rad_shock_steady" ) {
    rad_shock_steady_init( state, grid, pin );
  } else if ( problem_name == "rad_shock" ) {
    rad_shock_init( state, grid, pin );
  } else if ( problem_name == "marshak" ) {
    marshak_init( state, grid, pin );
  } else {
    THROW_ATHELAS_ERROR( " ! Please choose a valid problem_name" );
  }
}
