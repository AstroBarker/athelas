/**
 * File    :  driver.cpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Main driver routine
 **/

#include <algorithm> // std::min
#include <iostream>
#include <limits>
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
#include "rad_utilities.hpp"
#include "slope_limiter.hpp"
#include "state.hpp"
#include "timestepper.hpp"

int main( int argc, char *argv[] ) {
  // Check cmd line args
  if ( argc < 2 ) {
    THROW_ATHELAS_ERROR( "No input file passed! Do: ./main IN_FILE" );
  }

  signal( SIGSEGV, segfault_handler );
  signal( SIGABRT, segfault_handler );

  // load input deck
  ProblemIn pin( argv[1] );

  /* --- Problem Parameters --- */
  const std::string problem_name = pin.problem_name;

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

  const std::string BC = pin.BC;

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

    IdealGas eos( pin.ideal_gamma );

    if ( not Restart ) {
      // --- Initialize fields ---
      InitializeFields( &state, &Grid, &pin );

      ApplyBC( state.Get_uCF( ), &Grid, order, BC );
      if ( opts.do_rad ) {
        ApplyBC( state.Get_uCR( ), &Grid, order, BC );
      }
    }

    // --- Datastructure for modal basis ---
    ModalBasis Basis( pin.Basis, state.Get_uPF( ), &Grid, order, nNodes, nX,
                      nGuard );

    WriteBasis( &Basis, nGuard, Grid.Get_ihi( ), nNodes, order, problem_name );

    // --- Initialize timestepper ---
    TimeStepper SSPRK( &pin, Grid );

    SlopeLimiter S_Limiter( &Grid, &pin );

    // --- Limit the initial conditions ---
    S_Limiter.ApplySlopeLimiter( state.Get_uCF( ), &Grid, &Basis );

    // -- print run parameters  and initial condition ---
    PrintSimulationParameters( Grid, &pin, CFL );
    WriteState( &state, Grid, &S_Limiter, problem_name, t, order, 0,
                opts.do_rad );

    // --- Timer ---
    Kokkos::Timer timer_total;
    Kokkos::Timer timer_zone_cycles;
    Real zc_ws      = 0.0; // zone cycles / wall second
    Real time_cycle = 0.0;

    Real dt_init = 1.0e-16;
    dt           = dt_init;

    const Real dt_init_frac = pin.dt_init_frac;

    // --- Evolution loop ---
    const double nlim   = ( pin.nlim == -1 )
                              ? std::numeric_limits<double>::infinity( )
                              : pin.nlim;
    const int &i_print  = pin.ncycle_out; // std out
    const Real &dt_hdf5 = pin.dt_hdf5; // h5 out
    int iStep           = 0;
    int i_out           = 1; // output label, start 1
    std::cout << " ~ Step    t       dt       zone_cycles / wall_second\n"
              << std::endl;
    while ( t < t_end && iStep <= nlim ) {
      timer_zone_cycles.reset( );

      dt = std::min(
          compute_timestep( state.Get_uCF( ), &Grid, &eos, CFL, &opts ),
          dt * dt_init_frac );
      if ( t + dt > t_end ) {
        dt = t_end - t;
      }

      if ( !opts.do_rad ) {
        SSPRK.UpdateFluid( Compute_Increment_Explicit, dt, &state, Grid, &Basis,
                           &eos, &S_Limiter, opts );
      } else {
        // TODO: compile time swap operator splitting
        // SSPRK.UpdateFluid( Compute_Increment_Explicit, 0.5 * dt, &state,
        // Grid,
        //                    &Basis, &eos, &S_Limiter, opts );
        // SSPRK.UpdateRadiation( Compute_Increment_Explicit_Rad, dt, &state,
        // Grid,
        //                        &Basis, &eos, &S_Limiter, opts );
        // SSPRK.UpdateFluid( Compute_Increment_Explicit, 0.5 * dt, &state,
        // Grid,
        //                    &Basis, &eos, &S_Limiter, opts );

        try {
          SSPRK.UpdateRadHydro(
              Compute_Increment_Explicit, Compute_Increment_Explicit_Rad,
              ComputeIncrement_Fluid_Rad, ComputeIncrement_Rad_Source, dt,
              &state, Grid, &Basis, &eos, &S_Limiter, opts );
        } catch ( const AthelasError &e ) {
          std::cerr << e.what( ) << std::endl;
          return AthelasExitCodes::FAILURE;
        } catch ( const std::exception &e ) {
          std::cerr << "Library Error: " << e.what( ) << std::endl;
          return AthelasExitCodes::FAILURE;
        }
      }

#ifdef ATHELAS_DEBUG
      try {
        check_state( &state, Grid.Get_ihi( ), pin.do_rad );
      } catch ( const AthelasError &e ) {
        std::cerr << e.what( ) << std::endl;
        std::printf( "!!! Bad State found, writing _final_ output file ...\n" );
        WriteState( &state, Grid, &S_Limiter, problem_name, t, order, -1,
                    opts.do_rad );
        return AthelasExitCodes::FAILURE;
      }
#endif

      t += dt;
      time_cycle += timer_zone_cycles.seconds( );
      timer_zone_cycles.reset( );

      // Write state
      if ( t >= i_out * dt_hdf5 ) {
        WriteState( &state, Grid, &S_Limiter, problem_name, t, order, i_out,
                    opts.do_rad );
        i_out += 1;
      }

      if ( iStep % i_print == 0 ) {
        zc_ws = static_cast<Real>( i_print ) * nX / time_cycle;
        std::printf( " ~ %d %.5e %.5e %.5e \n", iStep, t, dt, zc_ws );
        time_cycle = 0.0;
      }

      iStep++;
    }

    // --- Finalize timer ---
    Real time = timer_total.seconds( );
    std::printf( " ~ Done! Elapsed time: %f seconds.\n", time );
    ApplyBC( state.Get_uCF( ), &Grid, order, BC );
    WriteState( &state, Grid, &S_Limiter, problem_name, t, order, -1,
                opts.do_rad );
  }
  Kokkos::finalize( );

  return AthelasExitCodes::SUCCESS;
}

/**
 * Pick number of quadrature points in order to evaluate polynomial of
 * at least order^2.
 * ! Broken for nNodes > order !
 **/
int NumNodes( const int order ) {
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

  const Real max_cfl = 0.95;
  return std::min( c * CFL / ( ( 2.0 * (order)-1.0 ) ), max_cfl );
}

/**
 * Compute timestep
 **/
Real compute_timestep( const View3D<Real> U, const GridStructure *Grid,
                       EOS *eos, const Real CFL, const Options *opts ) {
  Real dt;
  if ( !opts->do_rad ) {
    dt = ComputeTimestep_Fluid( U, Grid, eos, CFL );
  } else {
    dt = ComputeTimestep_Rad( Grid, CFL );
  }
  return dt;
}
