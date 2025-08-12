#include "driver.hpp"
#include "atom/atom.hpp"
#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/hydro_package.hpp"
#include "geometry/grid.hpp"
#include "gravity/gravity_package.hpp"
#include "history/quantities.hpp"
#include "initialization.hpp"
#include "interface/packages_base.hpp"
#include "io/io.hpp"
#include "limiters/slope_limiter.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "opacity/opac_variant.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/radhydro_package.hpp"
#include "state/state.hpp"
#include "timestepper/timestepper.hpp"
#include "utils/abstractions.hpp"
#include "utils/error.hpp"

auto Driver::execute() -> int {
  const auto nx           = pin_->param()->get<int>("problem.nx");
  const auto problem_name = pin_->param()->get<std::string>("problem.problem");


  // some startup io
  write_basis(fluid_basis_.get(), pin_->param()->get<int>("problem.nx"),
              pin_->param()->get<int>("fluid.nnodes"),
              pin_->param()->get<int>("fluid.porder"),
              pin_->param()->get<std::string>("problem.problem"));
  print_simulation_parameters(grid_, pin_.get());
  write_state(&state_, grid_, &sl_hydro_, problem_name, time_,
              pin_->param()->get<int>("fluid.porder"), 0, opts_.do_rad);

  // --- Timer ---
  Kokkos::Timer timer_zone_cycles;
  double zc_ws = 0.0; // zone cycles / wall second

  const double nlim       = (pin_->param()->get<double>("problem.nlim")) == -1
                                ? std::numeric_limits<double>::infinity()
                                : pin_->param()->get<double>("problem.nlim");
  const auto ncycle_out   = pin_->param()->get<int>("output.ncycle_out");
  const auto dt_init      = pin_->param()->get<double>("output.initial_dt");
  const auto dt_init_frac = pin_->param()->get<double>("output.dt_init_frac");
  const auto dt_hdf5      = pin_->param()->get<double>("output.dt_hdf5");

  dt_ = dt_init;

  // --- Evolution loop ---
  int iStep      = 0;
  int i_out_h5   = 1; // output label, start 1
  int i_out_hist = 1; // output hist
  std::println("# Step    t       dt       zone_cycles / wall_second");
  while (time_ < t_end_ && iStep <= nlim) {

    dt_ = std::min(manager_->min_timestep(
                       state_.u_cf(), grid_,
                       {.t = time_, .dt = dt_, .dt_a = 0.0, .stage = 0}),
                   dt_ * dt_init_frac);
    if (time_ + dt_ > t_end_) {
      dt_ = t_end_ - time_;
    }

    if (!opts_.do_rad) {
      ssprk_.step(manager_.get(), &state_, grid_, dt_, &sl_hydro_, &opts_);
    } else {
      try {
        ssprk_.step_imex(manager_.get(), &state_, grid_, dt_, &sl_hydro_,
                         &sl_rad_, &opts_);
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
      write_state(&state_, grid_, &sl_hydro_, problem_name, time_,
                  pin_->param()->get<int>("fluid.porder"), -1, opts_.do_rad);
      return AthelasExitCodes::FAILURE;
    }
#endif

    time_ += dt_;

    // Write state, other io
    if (time_ >= i_out_h5 * dt_hdf5) {
      write_state(&state_, grid_, &sl_hydro_, problem_name, time_,
                  fluid_basis_->get_order(), i_out_h5, opts_.do_rad);
      i_out_h5 += 1;
    }

    if (time_ >= i_out_hist * pin_->param()->get<double>("output.hist_dt")) {
      history_->write(state_, grid_, fluid_basis_.get(), radiation_basis_.get(),
                      time_);
      i_out_hist += 1;
    }

    // timer
    if (iStep % ncycle_out == 0) {
      zc_ws =
          static_cast<double>(ncycle_out) * nx / timer_zone_cycles.seconds();
      std::println("{} {:.5e} {:.5e} {:.5e}", iStep, time_, dt_, zc_ws);
      timer_zone_cycles.reset();
    }

    iStep++;
  }

  write_state(&state_, grid_, &sl_hydro_, problem_name, time_,
              pin_->param()->get<int>("fluid.porder"), -1, opts_.do_rad);

  return AthelasExitCodes::SUCCESS;
}

void Driver::initialize(ProblemIn* pin) { // NOLINT
  using fluid::HydroPackage;
  using gravity::GravityPackage;

  const auto nx = pin_->param()->get<int>("problem.nx");
  const int max_order =
      std::max(pin_->param()->get<int>("fluid.porder"),
               pin_->param()->get<int>("radiation.porder", 1));
  const auto cfl =
      compute_cfl(pin_->param()->get<double>("problem.cfl"), max_order);

  if (!restart_) {
    // The pattern here is annoying and due to a chicken-and-egg
    // pattern between problem generation and basis construction.
    // Some problems, like Shu-Osher, need the basis at setup
    // to perform the L2 projection from nodal to modal
    // representation. Basis construction, however, requires the
    // nodal density field as density weighted inner products are used.
    // So here, the firist initialize_fields call may only populate nodal
    // density in uPF. Then bases are constructed. Then, the second
    // initialize_fields call populates the conserved variables.
    // For simple cases, like Sod, the layering is redundant, as
    // the bases are never used.
    initialize_fields(&state_, &grid_, eos_.get(), pin);

    // --- Datastructure for modal basis ---
    fluid_basis_ = std::make_unique<ModalBasis>(
        poly_basis::poly_basis::legendre, state_.u_pf(), &grid_,
        pin->param()->get<int>("fluid.porder"),
        pin->param()->get<int>("fluid.nnodes"),
        pin->param()->get<int>("problem.nx"), true);
    if (opts_.do_rad) {
      radiation_basis_ = std::make_unique<ModalBasis>(
          poly_basis::poly_basis::legendre, state_.u_pf(), &grid_,
          pin->param()->get<int>("radiation.porder"),
          pin->param()->get<int>("radiation.nnodes"),
          pin->param()->get<int>("problem.nx"), false);
    }

    // --- Phase 2: Re-initialize with modal projection ---
    // This will use the nodal density from Phase 1 to construct proper modal
    // coefficients
    initialize_fields(&state_, &grid_, eos_.get(), pin, fluid_basis_.get(),
                      radiation_basis_.get());
  }

  const bool rad_active     = pin->param()->get<bool>("physics.rad_active");
  const bool gravity_active = pin->param()->get<bool>("physics.gravity_active");

  // --- Init physics package manager ---
  // NOTE: Hydro/RadHydro should be registered first
  if (rad_active) {
    manager_->add_package(RadHydroPackage{
        pin, ssprk_.get_n_stages(), eos_.get(), opac_.get(), fluid_basis_.get(),
        radiation_basis_.get(), bcs_.get(), cfl, nx, true});
  } else {
    [[unlikely]] // pure Hydro
    manager_->add_package(HydroPackage{pin, ssprk_.get_n_stages(), eos_.get(),
                                       fluid_basis_.get(), bcs_.get(), cfl, nx,
                                       true});
  }
  if (gravity_active) {
    manager_->add_package(
        GravityPackage{pin, pin->param()->get<GravityModel>("gravity.model"),
                       pin->param()->get<double>("gravity.gval"),
                       fluid_basis_.get(), cfl, true});
  }
  auto registered_pkgs = manager_->get_package_names();
  std::print("# Registered Packages ::");
  for (auto name : registered_pkgs) {
    std::print(" {}", name);
  }
  std::print("\n\n");

  // --- atomic data ---
  if (pin_->param()->get<bool>("physics.ionization_enabled")) {
    atomic_data_.emplace(
      pin_->param()->get<std::string>("ionization.fn_ionization"),
      pin_->param()->get<std::string>("ionization.fn_degeneracy"));
  }

  // --- slope limiter to initial condition ---
  apply_slope_limiter(&sl_hydro_, state_.u_cf(), &grid_, fluid_basis_.get(),
                      eos_.get());

  // --- Add history outputs ---
  // NOTE: Could be nice to have gravitational energy added
  // to total, conditionally.
  history_->add_quantity("Total Mass [g]", analysis::total_mass);
  history_->add_quantity("Total Fluid Energy [erg]",
                         analysis::total_fluid_energy);
  history_->add_quantity("Total Internal Energy [erg]",
                         analysis::total_internal_energy);
  history_->add_quantity("Total Kinetic Energy [erg]",
                         analysis::total_kinetic_energy);

  if (gravity_active) {
    history_->add_quantity("Total Gravitational Energy [erg]",
                           analysis::total_gravitational_energy);
  }

  if (rad_active) {
    history_->add_quantity("Total Radiation Momentum [g cm / s]",
                           analysis::total_rad_momentum);
    history_->add_quantity("Total Momentum [g cm / s]",
                           analysis::total_momentum);
    history_->add_quantity("Total Radiation Energy [erg]",
                           analysis::total_rad_energy);
    history_->add_quantity("Total Energy [erg]", analysis::total_energy);
  }
  history_->add_quantity("Total Fluid Momentum [g cm / s]",
                         analysis::total_fluid_momentum);
}

using limiter_utilities::initialize_slope_limiter;
// Driver
Driver::Driver(std::shared_ptr<ProblemIn> pin) // NOLINT
    : pin_(pin), manager_(std::make_unique<PackageManager>()),
      restart_(pin->param()->get<bool>("problem.restart")),
      bcs_(std::make_unique<BoundaryConditions>(
          bc::make_boundary_conditions(pin.get()))),
      time_(0.0), dt_(pin_->param()->get<double>("output.initial_dt")),
      t_end_(pin->param()->get<double>("problem.tf")),
      eos_(std::make_unique<EOS>(initialize_eos(pin.get()))),
      opac_(std::make_unique<Opacity>(initialize_opacity(pin.get()))),
      grid_(pin.get()),
      opts_(pin->param()->get<bool>("physics.rad_active"),
            pin->param()->get<bool>("physics.gravity_active"), restart_,
            pin->param()->get<int>("fluid.porder", 0)),
      state_(3 + 2 * (pin->param()->get<bool>("physics.rad_active")), 3, 1,
             pin->param()->get<int>("problem.nx"),
             pin->param()->get<int>("fluid.nnodes"),
             pin->param()->get<int>("fluid.porder"), 
             pin_->param()->get<bool>("physics.composition_enabled")), // TODO(astrobarker): thread in ncomps
      sl_hydro_(
          initialize_slope_limiter("fluid", &grid_, pin.get(), {0, 1, 2}, 3)),
      sl_rad_(initialize_slope_limiter("radiation", &grid_, pin.get(), {3, 4},
                                       2)), // update
      ssprk_(pin.get(), &grid_, eos_.get()),
      history_(std::make_unique<HistoryOutput>(
          pin->param()->get<std::string>("output.hist_fn"),
          pin->param()->get<bool>("output.history_enabled"))) {
  initialize(pin.get());
}
