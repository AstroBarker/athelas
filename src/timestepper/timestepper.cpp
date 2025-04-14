/**
 * @file timestepper.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Primary time marching routine
 */

#include <iostream>
#include <vector>

#include "error.hpp"
#include "fluid_discretization.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"
#include "slope_limiter.hpp"
#include "slope_limiter_base.hpp"
#include "tableau.hpp"
#include "timestepper.hpp"

/**
 * The constructor creates the necessary data structures for time evolution.
 * Lots of structures used in discretizations live here.
 **/
TimeStepper::TimeStepper( ProblemIn *pin, GridStructure &grid )
    : mSize_( grid.get_n_elements( ) + (2 * grid.get_guard( )) ),
      nStages_( pin->nStages ), tOrder_( pin->tOrder ),
      implicit_tableau_(
          ButcherTableau( nStages_, tOrder_, TableauType::Implicit ) ),
      explicit_tableau_(
          ButcherTableau( nStages_, tOrder_, TableauType::Explicit ) ),
      U_s_( "U_s", nStages_ + 1, 3, mSize_ + 1, pin->pOrder ),
      dU_s_( "dU_s", nStages_ + 1, 3, mSize_ + 1, pin->pOrder ),
      U_s_r_( "U_s", nStages_ + 1, 2, mSize_ + 1, pin->pOrder ),
      dU_s_r_( "dU_s", nStages_ + 1, 2, mSize_ + 1, pin->pOrder ),
      SumVar_U_( "SumVar_U", 3, mSize_ + 1, pin->pOrder ),
      SumVar_U_r_( "SumVar_U_r", 2, mSize_ + 1, pin->pOrder ),
      grid_s_( nStages_ + 1, GridStructure( pin ) ),
      stage_data_( "StageData", nStages_ + 1, mSize_ + 1 ),
      flux_q_( "flux_q_", 3, mSize_ + 1, grid.get_n_nodes( ) ),
      dFlux_num_( "Numerical Flux", 3, mSize_ + 1 ),
      uCF_F_L_( "Face L", 3, mSize_ ), uCF_F_R_( "Face R", 3, mSize_ ),
      flux_u_( "flux_u_", nStages_ + 1, mSize_ + 1 ),
      flux_p_( "flux_p_", mSize_ + 1 ) {}
