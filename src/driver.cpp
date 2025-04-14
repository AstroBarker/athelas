/**
 * @file driver.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief main driver routine
 *
 */

#include <algorithm> // std::min
#include <cmath>
#include <limits>
#include <print>
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
#include "io/io.hpp"
#include "opacity/opac.hpp"
#include "opacity/opac_base.hpp"
#include "opacity/opac_variant.hpp"
#include "problem_in.hpp"
#include "rad_discretization.hpp"
#include "rad_utilities.hpp"
#include "slope_limiter.hpp"
#include "slope_limiter_base.hpp"
#include "slope_limiter_utilities.hpp"
#include "state.hpp"
#include "timestepper.hpp"

namespace {

/**
 * Compute the CFL timestep restriction.
 **/
auto compute_cfl( const Real CFL, const int order, const int nStages,
                  const int tOrder ) -> Real {
  Real c = 1.0;

  if ( nStages == tOrder ) {
    c = 1.0;
  }
  if ( nStages != tOrder ) {
    if ( tOrder == 2 ) {
      c = 1.0;
    }
    if ( tOrder == 3 ) {
      c = 1.0;
    }
    if ( tOrder == 4 ) {
      c = 0.76;
    }
  }

  const Real max_cfl = 0.95;
  return std::min( c * CFL / ( ( 2.0 * (order)-1.0 ) ), max_cfl );
}

/**
 * Compute timestep
 **/
auto compute_timestep( const View3D<Real> U, const GridStructure* grid,
                       EOS* eos, const Real CFL, const Options* opts ) -> Real {
  Real dt = NAN;
  if ( !opts->do_rad ) {
    dt = fluid::compute_timestep_fluid( U, grid, eos, CFL );
  } else {
    dt = radiation::compute_timestep_rad( grid, CFL );
  }
  return dt;
}

} // namespace

auto main( int argc, char** argv ) -> int {
  // Check cmd line args
  if ( argc < 2 ) {
    THROW_ATHELAS_ERROR( "No input file passed! Do: ./main IN_FILE" );
  }

  auto sig1 = signal( SIGSEGV, segfault_handler );
  auto sig2 = signal( SIGABRT, segfault_handler );

  // create span of args
  auto args = std::span( argv, static_cast<size_t>( argc ) );

  // load input deck
  ProblemIn pin( args[1] );

  /* --- Problem Parameters --- */
  const std::string& problem_name = pin.problem_name;

  const int& nX      = pin.nElements;
  const int& order   = pin.pOrder;
  const int& nNodes  = pin.nNodes;
  const int& nStages = pin.nStages;
  const int& tOrder  = pin.tOrder;

  const int& nGuard = pin.nGhost;

  Real t           = 0.0;
  Real dt          = 0.0;
  const Real t_end = pin.t_end;

  const bool Restart = pin.Restart;

  const std::string BC = pin.BC;

  const Real CFL = compute_cfl( pin.CFL, order, nStages, tOrder );

  /* opts struct TODO: add grav when ready */
  Options opts = { .do_rad  = pin.do_rad,
                   .do_grav = false,
                   .restart = pin.Restart,
                   .BC      = BC,
                   .geom    = pin.Geometry,
                   .basis   = pin.basis };

  Kokkos::initialize( argc, argv );
  {

    // --- Create the grid object ---
    GridStructure grid( &pin );

    // --- Create the data structures ---
    const int nCF = 3;
    const int nPF = 3;
    const int nAF = 1;
    const int nCR = 2;
    State state( nCF, nCR, nPF, nAF, nX, nGuard, nNodes, order );

    IdealGas eos( pin.ideal_gamma );

    // opac
    Opacity opac = initialize_opacity( &pin );

    if ( not Restart ) {
      // --- Initialize fields ---
      initialize_fields( &state, &grid, &eos, &pin );

      bc::apply_bc( state.get_u_cf( ), &grid, order, BC );
      if ( opts.do_rad ) {
        bc::apply_bc( state.get_u_cr( ), &grid, order, BC );
      }
    }

    // --- Datastructure for modal basis ---
    ModalBasis basis( pin.basis, state.get_u_pf( ), &grid, order, nNodes, nX,
                      nGuard );

    write_basis( &basis, nGuard, grid.get_ihi( ), nNodes, order, problem_name );

    // --- Initialize timestepper ---
    TimeStepper SSPRK( &pin, grid );

    SlopeLimiter S_Limiter =
        limiter_utilities::initialize_slope_limiter( &grid, &pin, 3 );
    // --- Limit the initial conditions ---
    apply_slope_limiter( &S_Limiter, state.get_u_cf( ), &grid, &basis );

    // -- print run parameters  and initial condition ---
    print_simulation_parameters( grid, &pin, CFL );
    write_state( &state, grid, &S_Limiter, problem_name, t, order, 0,
                 opts.do_rad );

    // --- Timer ---
    Kokkos::Timer const timer_total;
    Kokkos::Timer timer_zone_cycles;
    Real zc_ws      = 0.0; // zone cycles / wall second
    Real time_cycle = 0.0;

    Real const dt_init = 1.0e-16;
    dt                 = dt_init;

    const Real dt_init_frac = pin.dt_init_frac;

    // --- Evolution loop ---
    const double nlim   = ( pin.nlim == -1 )
                              ? std::numeric_limits<double>::infinity( )
                              : pin.nlim;
    const int& i_print  = pin.ncycle_out; // std out
    const Real& dt_hdf5 = pin.dt_hdf5; // h5 out
    int iStep           = 0;
    int i_out           = 1; // output label, start 1
    std::cout << " ~ Step    t       dt       zone_cycles / wall_second\n"
              << std::endl;
    while ( t < t_end && iStep <= nlim ) {
      timer_zone_cycles.reset( );

      dt = std::min(
          compute_timestep( state.get_u_cf( ), &grid, &eos, CFL, &opts ),
          dt * dt_init_frac );
      if ( t + dt > t_end ) {
        dt = t_end - t;
      }

      if ( !opts.do_rad ) {
        SSPRK.update_fluid( fluid::compute_increment_explicit, dt, &state, grid,
                            &basis, &eos, &S_Limiter, &opts );
      } else {
        // TODO(astrobarker): compile time swap operator splitting
        // SSPRK.update_fluid( compute_increment_explicit, 0.5 * dt, &state,
        // grid,
        //                    &basis, &eos, &S_Limiter, opts );
        // SSPRK.update_radiation( compute_increment_explicit_rad, dt, &state,
        // grid,
        //                        &basis, &eos, &S_Limiter, opts );
        // SSPRK.update_fluid( compute_increment_explicit, 0.5 * dt, &state,
        // grid,
        //                    &basis, &eos, &S_Limiter, opts );

        try {
          SSPRK.update_rad_hydro( fluid::compute_increment_explicit,
                                  radiation::compute_increment_explicit_rad,
                                  fluid::compute_increment_fluid_rad,
                                  radiation::compute_increment_rad_source, dt,
                                  &state, grid, &basis, &eos, &opac, &S_Limiter,
                                  &opts );
        } catch ( const AthelasError& e ) {
          std::cerr << e.what( ) << std::endl;
          return AthelasExitCodes::FAILURE;
        } catch ( const std::exception& e ) {
          std::cerr << "Library Error: " << e.what( ) << std::endl;
          return AthelasExitCodes::FAILURE;
        }
      }

#ifdef ATHELAS_DEBUG
      try {
        check_state( &state, grid.get_ihi( ), pin.do_rad );
      } catch ( const AthelasError& e ) {
        std::cerr << e.what( ) << std::endl;
        std::println( "!!! Bad State found, writing _final_ output file ..." );
        write_state( &state, grid, &S_Limiter, problem_name, t, order, -1,
                     opts.do_rad );
        return AthelasExitCodes::FAILURE;
      }
#endif

      t += dt;
      time_cycle += timer_zone_cycles.seconds( );
      timer_zone_cycles.reset( );

      // Write state
      if ( t >= i_out * dt_hdf5 ) {
        write_state( &state, grid, &S_Limiter, problem_name, t, order, i_out,
                     opts.do_rad );
        i_out += 1;
      }

      // timer
      if ( iStep % i_print == 0 ) {
        zc_ws = static_cast<Real>( i_print ) * nX / time_cycle;
        std::println( " ~ {} {:.5e} {:.5e} {:.5e} ", iStep, t, dt, zc_ws );
      }

      iStep++;
    }

    // --- Finalize timer ---
    Real const time = timer_total.seconds( );
    std::println( " ~ Done! Elapsed time: {} seconds.", time );
    bc::apply_bc( state.get_u_cf( ), &grid, order, BC );
    write_state( &state, grid, &S_Limiter, problem_name, t, order, -1,
                 opts.do_rad );
  }
  Kokkos::finalize( );

  return AthelasExitCodes::SUCCESS;
}
