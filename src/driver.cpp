/**
 * @file driver.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief main driver routine
 *
 */

#include "driver.hpp"
#include "abstractions.hpp"
#include "bc/boundary_conditions.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "build_info.hpp"
#include "eos.hpp"
#include "error.hpp"
#include "fluid_discretization.hpp"
#include "fluid_utilities.hpp"
#include "grid.hpp"
#include "initialization.hpp"
#include "io/io.hpp"
#include "main.hpp"
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
  Real dt = 0.0;
  if ( !opts->do_rad ) {
    dt = fluid::compute_timestep_fluid( U, grid, eos, CFL );
  } else {
    dt = radiation::compute_timestep_rad( grid, CFL );
  }
  return dt;
}

} // namespace

void Driver::initialize( const ProblemIn* pin ) { // NOLINT
  if ( !restart_ ) {
    // --- Initialize fields ---
    initialize_fields( &state_, &grid_, &eos_, pin );

    fill_ghost_zones<3>( state_.get_u_cf( ), &grid_, pin->pOrder, bcs_.get( ) );
    if ( opts_.do_rad ) {
      bc::fill_ghost_zones<2>( state_.get_u_cr( ), &grid_, pin->pOrder,
                               bcs_.get( ) );
    }
  }

  // --- Datastructure for modal basis ---
  basis_ = std::make_unique<ModalBasis>( pin->basis, state_.get_u_pf( ), &grid_,
                                         pin->pOrder, pin->nNodes,
                                         pin->nElements, pin->nGhost );

  // --- slope limiter to initial condition ---
  apply_slope_limiter( &sl_hydro_, state_.get_u_cf( ), &grid_, basis_.get( ),
                       &eos_ );
}

using limiter_utilities::initialize_slope_limiter;
// Driver
Driver::Driver( const ProblemIn* pin ) // NOLINT
    : pin_( *pin ), nX_( pin->nElements ), problem_name_( pin->problem_name ),
      restart_( pin->Restart ),
      bcs_( std::make_unique<BoundaryConditions>( bc::make_boundary_conditions(
          pin->do_rad, pin->fluid_bc_i, pin->fluid_bc_o,
          pin->fluid_i_dirichlet_values, pin->fluid_o_dirichlet_values,
          pin->rad_bc_i, pin->rad_bc_o, pin->rad_i_dirichlet_values,
          pin->rad_o_dirichlet_values ) ) ),
      time_( 0.0 ), dt_( 1.0e-16 ), t_end_( pin->t_end ),
      cfl_( compute_cfl( pin->CFL, pin->pOrder, pin->nStages, pin->tOrder ) ),
      i_print_( pin->ncycle_out ),
      nlim_( ( pin->nlim == -1 ) ? std::numeric_limits<double>::infinity( )
                                 : pin->nlim ),
      dt_hdf5_( pin->dt_hdf5 ), dt_init_frac_( pin->dt_init_frac ),
      eos_( IdealGas( pin->gamma_eos ) ), opac_( initialize_opacity( pin ) ),
      grid_( pin ),
      opts_( pin->do_rad, false, restart_, pin->Geometry, pin->basis ),
      state_( 3, 2, 3, 1, pin->nElements, pin->nGhost, pin->nNodes,
              pin->pOrder ),
      sl_hydro_( initialize_slope_limiter( &grid_, pin, 3 ) ),
      sl_rad_( initialize_slope_limiter( &grid_, pin, 2 ) ), // update
      ssprk_( pin, &grid_ ) {
  initialize( pin );
}

auto Driver::execute( ) -> int {
  // some startup io
  write_basis( basis_.get( ), pin_.nGhost, pin_.nElements, pin_.nNodes,
               pin_.pOrder, pin_.problem_name );
  print_simulation_parameters( grid_, &pin_, cfl_ );
  write_state( &state_, grid_, &sl_hydro_, problem_name_, time_, pin_.pOrder, 0,
               opts_.do_rad );

  // --- Timer ---
  Kokkos::Timer timer_zone_cycles;
  Real zc_ws = 0.0; // zone cycles / wall second

  // initial timestep TODO(astrobarker) make input param
  Real const dt_init = 1.0e-16;
  dt_                = dt_init;

  // --- Evolution loop ---
  int iStep = 0;
  int i_out = 1; // output label, start 1
  std::cout << " ~ Step    t       dt       zone_cycles / wall_second\n"
            << std::endl;
  while ( time_ < t_end_ && iStep <= nlim_ ) {
    timer_zone_cycles.reset( );

    dt_ = std::min(
        compute_timestep( state_.get_u_cf( ), &grid_, &eos_, cfl_, &opts_ ),
        dt_ * dt_init_frac_ );
    if ( time_ + dt_ > t_end_ ) {
      dt_ = t_end_ - time_;
    }

    if ( !opts_.do_rad ) {
      ssprk_.update_fluid( dt_, &state_, grid_, basis_.get( ), &eos_,
                           &sl_hydro_, &opts_, bcs_.get( ) );
    } else {
      try {
        ssprk_.update_rad_hydro( dt_, &state_, grid_, basis_.get( ), &eos_,
                                 &opac_, &sl_hydro_, &opts_, bcs_.get( ) );
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
      check_state( &state_, grid_.get_ihi( ), opts_.do_rad );
    } catch ( const AthelasError& e ) {
      std::cerr << e.what( ) << std::endl;
      std::println( "!!! Bad State found, writing _final_ output file ..." );
      write_state( &state_, grid_, &sl_hydro_, problem_name_, time_,
                   pin_.pOrder, -1, opts_.do_rad );
      return AthelasExitCodes::FAILURE;
    }
#endif

    time_ += dt_;

    // Write state
    if ( time_ >= i_out * dt_hdf5_ ) {
      write_state( &state_, grid_, &sl_hydro_, problem_name_, time_,
                   basis_.get( )->get_order( ), i_out, opts_.do_rad );
      i_out += 1;
    }

    // timer
    if ( iStep % i_print_ == 0 ) {
      zc_ws =
          static_cast<Real>( i_print_ ) * nX_ / timer_zone_cycles.seconds( );
      std::println( " ~ {} {:.5e} {:.5e} {:.5e} ", iStep, time_, dt_, zc_ws );
      timer_zone_cycles.reset( );
    }

    iStep++;
  }

  // --- Apply bc and write final output ---
  bc::fill_ghost_zones<3>( state_.get_u_cf( ), &grid_, pin_.pOrder,
                           bcs_.get( ) );
  write_state( &state_, grid_, &sl_hydro_, problem_name_, time_, pin_.pOrder,
               -1, opts_.do_rad );

  return AthelasExitCodes::SUCCESS;
}
