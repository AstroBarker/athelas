/**
 * File     :  timestepper.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : SSPRK timestepping routines
 **/

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
TimeStepper::TimeStepper( ProblemIn *pin, GridStructure &Grid )
    : mSize( Grid.Get_nElements( ) + 2 * Grid.Get_Guard( ) ),
      nStages( pin->nStages ), tOrder( pin->tOrder ), BC( pin->BC ),
      implicit_tableau_(
          ButcherTableau( nStages, tOrder, TableauType::Implicit ) ),
      explicit_tableau_(
          ButcherTableau( nStages, tOrder, TableauType::Explicit ) ),
      U_s( "U_s", nStages + 1, 3, mSize + 1, pin->pOrder ),
      dU_s( "dU_s", nStages + 1, 3, mSize + 1, pin->pOrder ),
      U_s_r( "U_s", nStages + 1, 2, mSize + 1, pin->pOrder ),
      dU_s_r( "dU_s", nStages + 1, 2, mSize + 1, pin->pOrder ),
      SumVar_U( "SumVar_U", 3, mSize + 1, pin->pOrder ),
      SumVar_U_r( "SumVar_U_r", 2, mSize + 1, pin->pOrder ),
      Grid_s( nStages + 1, GridStructure( pin ) ),
      StageData( "StageData", nStages + 1, mSize + 1 ),
      Flux_q( "Flux_q", 3, mSize + 1, Grid.Get_nNodes( ) ),
      dFlux_num( "Numerical Flux", 3, mSize + 1 ),
      uCF_F_L( "Face L", 3, mSize ), uCF_F_R( "Face R", 3, mSize ),
      Flux_U( "Flux_U", nStages + 1, mSize + 1 ),
      Flux_P( "Flux_P", mSize + 1 ) {}
