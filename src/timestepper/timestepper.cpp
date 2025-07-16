/**
 * @file timestepper.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Primary time marching routine
 */

#include <vector>

#include "grid.hpp"
#include "tableau.hpp"
#include "timestepper.hpp"

/**
 * The constructor creates the necessary data structures for time evolution.
 * Lots of structures used in discretizations live here.
 **/
TimeStepper::TimeStepper( const ProblemIn* pin, GridStructure* grid )
    : mSize_( grid->get_n_elements( ) + ( 2 * grid->get_guard( ) ) ),
      integrator_(create_tableau(pin->method_id)),
      nStages_( integrator_.num_stages ), tOrder_( integrator_.explicit_order ),
      U_s_( "U_s", nStages_ + 1, 3, mSize_ + 1, pin->pOrder ),
      dU_s_( "dU_s", nStages_ + 1, 3, mSize_ + 1, pin->pOrder ),
      U_s_r_( "U_s_r", nStages_ + 1, 2, mSize_ + 1, pin->pOrder ),
      dU_s_r_( "dU_s", nStages_ + 1, 2, mSize_ + 1, pin->pOrder ),
      SumVar_U_( "SumVar_U", 3, mSize_ + 1, pin->pOrder ),
      SumVar_U_r_( "SumVar_U_r", 2, mSize_ + 1, pin->pOrder ),
      grid_s_( nStages_ + 1, GridStructure( pin ) ),
      stage_data_( "StageData", nStages_ + 1, mSize_ + 1 ),
      dFlux_num_( "Numerical Flux", 3, mSize_ + 1 ),
      uCF_F_L_( "Face L", 3, mSize_ ), uCF_F_R_( "Face R", 3, mSize_ ),
      flux_u_( "flux_u_", nStages_ + 1, mSize_ + 1 ) {}
