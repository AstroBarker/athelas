#pragma once

#include <memory>

#include "atom/atom.hpp"
#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "history/history.hpp"
#include "interface/packages_base.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "pgen/problem_in.hpp"
#include "timestepper/timestepper.hpp"

namespace athelas {

using atom::AtomicData;
using bc::BoundaryConditions;
using limiter_utilities::initialize_slope_limiter;

/**
 * @class Driver
 * @brief the primary executor of the simulation.
 * Owns key data and calls timestepper, IO.
 */
class Driver {
 public:
  //  explicit Driver(std::shared_ptr<ProblemIn> pin);
  // Driver
  explicit Driver(std::shared_ptr<ProblemIn> pin) // NOLINT
      : pin_(pin), manager_(std::make_unique<PackageManager>()),
        restart_(pin->param()->get<bool>("problem.restart")),
        bcs_(std::make_unique<BoundaryConditions>(
            bc::make_boundary_conditions(pin.get()))),
        time_(0.0), dt_(pin_->param()->get<double>("output.initial_dt")),
        t_end_(pin->param()->get<double>("problem.tf")),
        eos_(std::make_unique<EOS>(initialize_eos(pin.get()))),
        opac_(std::make_unique<Opacity>(initialize_opacity(pin.get()))),
        grid_(pin.get()), sl_hydro_(initialize_slope_limiter(
                              "fluid", &grid_, pin.get(), {0, 1, 2}, 3)),
        sl_rad_(initialize_slope_limiter("radiation", &grid_, pin.get(), {3, 4},
                                         2)), // update
        ssprk_(pin.get(), &grid_, eos_.get()),
        history_(std::make_unique<HistoryOutput>(
            pin->param()->get<std::string>("output.hist_fn"),
            pin->param()->get<bool>("output.history_enabled"))) {
    static const bool rad_enabled =
        pin->param()->get<bool>("physics.rad_active");
    static const bool composition_enabled =
        pin->param()->get<bool>("physics.composition_enabled");
    static const bool ionization_enabled =
        pin->param()->get<bool>("physics.ionization_enabled");
    static const int nvars_cons = (rad_enabled) ? 5 : 3;
    static const int nvars_prim = 3; // Maybe this can be smarter
    static const int nvars_aux = (rad_enabled) ? 5 : 3;
    static const int n_stages = ssprk_.n_stages();
    static const int nx = pin->param()->get<int>("problem.nx");
    static const int n_nodes = pin->param()->get<int>("fluid.nnodes");
    static const int porder = pin->param()->get<int>("fluid.porder");
    state_ = std::make_unique<State>(nvars_cons, nvars_prim, nvars_aux, nx,
                                     n_nodes, porder, n_stages,
                                     composition_enabled, ionization_enabled);
    initialize(pin.get());
  }

  auto execute() -> int;

 private:
  // init
  void initialize(ProblemIn *pin);

  std::shared_ptr<ProblemIn> pin_;

  std::unique_ptr<PackageManager> manager_;

  // TODO(astrobarker): thread in run_id_
  // std::string run_id_;
  bool restart_;

  std::unique_ptr<BoundaryConditions> bcs_;

  double time_{};
  double dt_;
  double t_end_;

  // core bits
  // TODO(astrobarker): keep eos_, opac_ in packages.
  std::unique_ptr<EOS> eos_;
  std::unique_ptr<Opacity> opac_;
  GridStructure grid_;

  // slope limiters
  SlopeLimiter sl_hydro_;
  SlopeLimiter sl_rad_;

  // timestepper
  TimeStepper ssprk_;

  // history
  std::unique_ptr<HistoryOutput> history_;

  // bases
  std::unique_ptr<ModalBasis> fluid_basis_; // init in constr body
  std::unique_ptr<ModalBasis> radiation_basis_; // init in constr body

  std::unique_ptr<State> state_;
}; // class Driver

} // namespace athelas

namespace {

/**
 * Compute the CFL timestep restriction.
 **/
KOKKOS_INLINE_FUNCTION
auto compute_cfl(const double CFL, const int order) -> double {
  double c = 1.0;

  const double max_cfl = 0.95;
  return std::min(c * CFL / ((2.0 * (order)-1.0)), max_cfl);
}
} // namespace
