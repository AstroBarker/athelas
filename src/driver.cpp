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
#include "basis/polynomial_basis.hpp"
#include "eos_variant.hpp"
#include "error.hpp"
#include "fluid/fluid_utilities.hpp"
#include "fluid/hydro_package.hpp"
#include "grid.hpp"
#include "initialization.hpp"
#include "io/io.hpp"
#include "opacity/opac_variant.hpp"
#include "problem_in.hpp"
#include "rad_utilities.hpp"
#include "slope_limiter.hpp"
#include "slope_limiter_utilities.hpp"
#include "state.hpp"
#include "timestepper.hpp"

namespace {

/**
 * Compute the CFL timestep restriction.
 **/
auto compute_cfl(const double CFL, const int order, const int nStages,
                 const int tOrder) -> double {
  double c = 1.0;

  if (nStages == tOrder) {
    c = 1.0;
  }
  if (nStages != tOrder) {
    if (tOrder == 2) {
      c = 1.0;
    }
    if (tOrder == 3) {
      c = 1.0;
    }
    if (tOrder == 4) {
      c = 0.76;
    }
  }

  const double max_cfl = 0.95;
  return std::min(c * CFL / ((2.0 * (order)-1.0)), max_cfl);
}

/**
 * Compute timestep
 **/
auto compute_timestep(const View3D<double> U, const GridStructure* grid,
                      EOS* eos, const double CFL, const Options* opts)
    -> double {
  double dt = 0.0;
  if (!opts->do_rad) {
    dt = fluid::compute_timestep_fluid(U, grid, eos, CFL);
  } else {
    dt = radiation::compute_timestep_rad(grid, CFL);
  }
  return dt;
}

} // namespace

void Driver::initialize(const ProblemIn* pin) { // NOLINT
  using fluid::HydroPackage;
  if (!restart_) {
    // --- Initialize fields ---
    initialize_fields(&state_, &grid_, eos_.get(), pin);

    // fill_ghost_zones<3>( state_.get_u_cf( ), &grid_, pin->pOrder, bcs_.get( )
    // ); if ( opts_.do_rad ) {
    //   bc::fill_ghost_zones<2>( state_.get_u_cr( ), &grid_, pin->pOrder,
    //                            bcs_.get( ) );
    // }
  }

  // --- Datastructure for modal basis ---
  fluid_basis_ = std::make_unique<ModalBasis>(
      pin->basis, state_.get_u_pf(), &grid_, pin->pOrder, pin->nNodes,
      pin->nElements, pin->nGhost, true);
  if (opts_.do_rad) {
    radiation_basis_ = std::make_unique<ModalBasis>(
        pin->basis, state_.get_u_pf(), &grid_, pin->pOrder, pin->nNodes,
        pin->nElements, pin->nGhost, false);
  }

  // --- Init physics package manager ---
  // TODO(astrobarker) package logic
  manager_->add_package(HydroPackage{pin, ssprk_.get_n_stages(), eos_.get(),
                                     fluid_basis_.get(), bcs_.get(), cfl_, nX_,
                                     true});

  // --- slope limiter to initial condition ---
  apply_slope_limiter(&sl_hydro_, state_.get_u_cf(), &grid_, fluid_basis_.get(),
                      eos_.get());
}

using limiter_utilities::initialize_slope_limiter;
// Driver
Driver::Driver(const ProblemIn* pin) // NOLINT
    : pin_(*pin), manager_(std::make_unique<PackageManager>()),
      nX_(pin->nElements), problem_name_(pin->problem_name),
      restart_(pin->Restart),
      bcs_(std::make_unique<BoundaryConditions>(bc::make_boundary_conditions(
          pin->do_rad, pin->fluid_bc_i, pin->fluid_bc_o,
          pin->fluid_i_dirichlet_values, pin->fluid_o_dirichlet_values,
          pin->rad_bc_i, pin->rad_bc_o, pin->rad_i_dirichlet_values,
          pin->rad_o_dirichlet_values))),
      time_(0.0), dt_(1.0e-16), t_end_(pin->t_end),
      cfl_(compute_cfl(pin->CFL, pin->pOrder, pin->nStages, pin->tOrder)),
      i_print_(pin->ncycle_out),
      nlim_((pin->nlim == -1) ? std::numeric_limits<double>::infinity()
                              : pin->nlim),
      dt_hdf5_(pin->dt_hdf5), dt_init_frac_(pin->dt_init_frac),
      eos_(std::make_unique<EOS>(initialize_eos(pin))),
      opac_(std::make_unique<Opacity>(initialize_opacity(pin))), grid_(pin),
      opts_(pin->do_rad, false, restart_, pin->Geometry, pin->basis),
      state_(3, 2, 3, 1, pin->nElements, pin->nGhost, pin->nNodes, pin->pOrder),
      sl_hydro_(initialize_slope_limiter(&grid_, pin, 3)),
      sl_rad_(initialize_slope_limiter(&grid_, pin, 2)), // update
      ssprk_(pin, &grid_) {
  initialize(pin);
}

auto Driver::execute() -> int {
  // some startup io
  write_basis(fluid_basis_.get(), pin_.nGhost, pin_.nElements, pin_.nNodes,
              pin_.pOrder, pin_.problem_name);
  print_simulation_parameters(grid_, &pin_, cfl_);
  write_state(&state_, grid_, &sl_hydro_, problem_name_, time_, pin_.pOrder, 0,
              opts_.do_rad);

  // --- Timer ---
  Kokkos::Timer timer_zone_cycles;
  double zc_ws = 0.0; // zone cycles / wall second

  // initial timestep TODO(astrobarker) make input param
  double const dt_init = 1.0e-16;
  dt_                  = dt_init;

  // --- Evolution loop ---
  int iStep = 0;
  int i_out = 1; // output label, start 1
  std::println("# Step    t       dt       zone_cycles / wall_second");
  while (time_ < t_end_ && iStep <= nlim_) {

    dt_ = std::min(
        compute_timestep(state_.get_u_cf(), &grid_, eos_.get(), cfl_, &opts_),
        dt_ * dt_init_frac_);
    if (time_ + dt_ > t_end_) {
      dt_ = t_end_ - time_;
    }

    if (!opts_.do_rad) {
      ssprk_.step(manager_.get(), &state_, grid_, dt_,
                          &sl_hydro_, &opts_);
    } else {
      try {
        ssprk_.update_rad_hydro(dt_, &state_, grid_, fluid_basis_.get(),
                                radiation_basis_.get(), eos_.get(), opac_.get(),
                                &sl_hydro_, &opts_, bcs_.get());
      } catch (const AthelasError& e) {
        std::cerr << e.what() << "\n";
        return AthelasExitCodes::FAILURE;
      } catch (const std::exception& e) {
        std::cerr << "Library Error: " << e.what() << "\n";
        return AthelasExitCodes::FAILURE;
      }
    }

#ifdef ATHELAS_DEBUG
    try {
      check_state(&state_, grid_.get_ihi(), opts_.do_rad);
    } catch (const AthelasError& e) {
      std::cerr << e.what() << std::endl;
      std::println("!!! Bad State found, writing _final_ output file ...");
      write_state(&state_, grid_, &sl_hydro_, problem_name_, time_, pin_.pOrder,
                  -1, opts_.do_rad);
      return AthelasExitCodes::FAILURE;
    }
#endif

    time_ += dt_;

    // Write state
    if (time_ >= i_out * dt_hdf5_) {
      write_state(&state_, grid_, &sl_hydro_, problem_name_, time_,
                  fluid_basis_.get()->get_order(), i_out, opts_.do_rad);
      i_out += 1;
    }

    // timer
    if (iStep % i_print_ == 0) {
      zc_ws = static_cast<double>(i_print_) * nX_ / timer_zone_cycles.seconds();
      std::println("{} {:.5e} {:.5e} {:.5e}", iStep, time_, dt_, zc_ws);
      timer_zone_cycles.reset();
    }

    iStep++;
  }

  // --- Apply bc and write final output ---
  // bc::fill_ghost_zones<3>( state_.get_u_cf( ), &grid_, fluid_basis_.get(),
  //                         bcs_.get( ) );
  write_state(&state_, grid_, &sl_hydro_, problem_name_, time_, pin_.pOrder, -1,
              opts_.do_rad);

  return AthelasExitCodes::SUCCESS;
}
