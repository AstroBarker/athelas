#pragma once

#include <memory>

#include "atom/atom.hpp"
#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "history/history.hpp"
#include "interface/packages_base.hpp"
#include "pgen/problem_in.hpp"
#include "timestepper/timestepper.hpp"

using atom::AtomicData;
using bc::BoundaryConditions;

/**
 * @class Driver
 * @brief the primary executor of the simulation.
 * Owns key data and calls timestepper, IO.
 */
class Driver {
 public:
  explicit Driver(std::shared_ptr<ProblemIn> pin);
  // explicit Driver(ProblemIn* pin);

  auto execute() -> int;

 private:
  // init
  void initialize(ProblemIn* pin);

  std::shared_ptr<ProblemIn> pin_;

  std::unique_ptr<PackageManager> manager_;

  // TODO(astrobarker): thread in run_id_
  // std::string run_id_;
  bool restart_;

  std::unique_ptr<BoundaryConditions> bcs_;

  double time_;
  double dt_;
  double t_end_;

  // core bits
  // TODO(astrobarker): kepe eos_, opac_ in packages.
  std::unique_ptr<EOS> eos_;
  std::unique_ptr<Opacity> opac_;
  GridStructure grid_;
  Options opts_;
  State state_;

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

  // The rest
  std::optional<AtomicData> atomic_data_;
}; // class Driver
//
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
