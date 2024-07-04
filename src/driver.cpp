/**
 * File    :  driver.cpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Main driver routine
 **/

#include <algorithm> // std::min
#include <iostream>
#include <string>
#include <vector>

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "boundary_conditions.hpp"
#include "driver.hpp"
#include "eos.hpp"
#include "error.hpp"
#include "fluid_discretization.hpp"
#include "fluid_utilities.hpp"
#include "grid.hpp"
#include "initialization.hpp"
#include "io.hpp"
#include "problem_in.hpp"
#include "rad_discretization.hpp"
#include "slope_limiter.hpp"
#include "state.hpp"
#include "timestepper.hpp"

int main( int argc, char *argv[] ) {
  // Check cmd line args
  if ( argc < 2 ) {
    throw Error( "No input file passed! Do: ./main IN_FILE" );
  }

  ProblemIn pin( argv[1] );

  /* --- Problem Parameters --- */
  const std::string ProblemName = pin.ProblemName;

  const int &nX      = pin.nElements;
  const int &order   = pin.pOrder;
  const int &nNodes  = pin.nNodes;
  const int &nStages = pin.nStages;
  const int &tOrder  = pin.tOrder;

  const int &nGuard = pin.nGhost;

  Real t           = 0.0;
  Real dt          = 0.0;
  const Real t_end = pin.t_end;

  const bool Restart = pin.Restart;

  const std::string BC   = pin.BC;
  const Real gamma_ideal = 1.4;

  const Real CFL = ComputeCFL( pin.CFL, order, nStages, tOrder );

  /* opts struct TODO: add grav when ready */
  Options opts = { pin.do_rad, false,        pin.Restart,
                   BC,         pin.Geometry, pin.Basis };

  Kokkos::initialize( argc, argv );
  {

    // --- Create the Grid object ---
    GridStructure Grid( &pin );

    // --- Create the data structures ---
    const int nCF = 3;
    const int nPF = 3;
    const int nAF = 1;
    const int nCR = 2;
    State state( nCF, nCR, nPF, nAF, nX, nGuard, nNodes, order );

    IdealGas eos( gamma_ideal );

    if ( not Restart ) {
      // --- Initialize fields ---
      InitializeFields( &state, &Grid, ProblemName );

      ApplyBC( state.Get_uCF( ), &Grid, order, BC );
    }

    // --- Datastructure for modal basis ---
    ModalBasis Basis( pin.Basis, state.Get_uPF( ), &Grid, order, nNodes, nX,
                      nGuard );

    WriteBasis( &Basis, nGuard, Grid.Get_ihi( ), nNodes, order, ProblemName );

    // --- Initialize timestepper ---
    TimeStepper SSPRK( &pin, Grid );

    SlopeLimiter S_Limiter( &Grid, &pin );

    // --- Limit the initial conditions ---
    S_Limiter.ApplySlopeLimiter( state.Get_uCF( ), &Grid, &Basis );

    // -- print run parameters  and initial condition ---
    PrintSimulationParameters( Grid, &pin, CFL );
    WriteState( &state, Grid, &S_Limiter, ProblemName, t, order, 0 );

    // --- Timer ---
    Kokkos::Timer timer;

    // --- Evolution loop ---
    const int i_print = 1000; // std out
    const int i_write = 1000; // h5 out
    int iStep         = 0;
    int i_out         = 1; // output label, start 1
    std::cout << " ~ Step\tt\tdt" << std::endl;
    while ( t < t_end ) {

      // TODO: ComputeTimestep_Rad
      dt = ComputeTimestep_Fluid( state.Get_uCF( ), &Grid, &eos, CFL );
      if ( opts.do_rad ) { // hack
        dt = std::pow( 10.0, -15.5 );
      }

      if ( t + dt > t_end ) {
        dt = t_end - t;
      }

      if ( iStep % i_print == 0 ) {
        std::printf( " ~ %d \t %.5e \t %.5e\n", iStep, t, dt );
      }

      if ( !opts.do_rad ) {
        SSPRK.UpdateFluid( Compute_Increment_Explicit, dt, &state, Grid, &Basis,
                           &eos, &S_Limiter, opts );
      } else {
        SSPRK.UpdateFluid( Compute_Increment_Explicit, 0.5 * dt, &state, Grid,
                           &Basis, &eos, &S_Limiter, opts );
        SSPRK.UpdateRadiation( Compute_Increment_Explicit_Rad, dt, &state, Grid,
                               &Basis, &eos, &S_Limiter, opts );
        SSPRK.UpdateFluid( Compute_Increment_Explicit, 0.5 * dt, &state, Grid,
                           &Basis, &eos, &S_Limiter, opts );

        // SSPRK.UpdateRadHydro( Compute_Increment_Explicit,
        //                       Compute_Increment_Explicit_Rad,
        //                       Compute_Increment_Explicit_Rad,
        //                       Compute_Increment_Explicit_Rad,
        //                       dt, &state, Grid,
        //                       &Basis, &eos, &S_Limiter, opts );
      }

#ifdef ATHELAS_DEBUG
      check_state( &state, Grid.Get_ihi( ), pin.do_rad );
#endif

      t += dt;

      // Write state
      if ( iStep % i_write == 0 ) {
        WriteState( &state, Grid, &S_Limiter, ProblemName, t, order, i_out );
        i_out += 1;
      }

      iStep++;
    }

    // --- Finalize timer ---
    Real time = timer.seconds( );
    std::printf( " ~ Done! Elapsed time: %f seconds.\n", time );
    ApplyBC( state.Get_uCF( ), &Grid, order, BC );
    WriteState( &state, Grid, &S_Limiter, ProblemName, t, order, -1 );
  }
  Kokkos::finalize( );

  return 0;
}

/**
 * Pick number of quadrature points in order to evaluate polynomial of
 * at least order^2.
 * ! Broken for nNodes > order !
 **/
int NumNodes( int order ) {
  if ( order <= 4 ) {
    return order;
  } else {
    return order + 1;
  }
}

/**
 * Compute the CFL timestep restriction.
 **/
Real ComputeCFL( const Real CFL, const int order, const int nStages,
                 const int tOrder ) {
  Real c = 1.0;

  if ( nStages == tOrder ) c = 1.0;
  if ( nStages != tOrder ) {
    if ( tOrder == 2 ) c = 1.0;
    if ( tOrder == 3 ) c = 1.0;
    if ( tOrder == 4 ) c = 0.76;
  }

  const Real max_cfl = 0.9;
  return std::min( c * CFL / ( ( 2.0 * (order)-1.0 ) ), max_cfl );
}
