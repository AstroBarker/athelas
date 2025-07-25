#pragma once
/**
 * @file driver.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Driver
 *
 * @details Functions:
 *            - NumNodes
 *            - compute_cfl
 *            - compute_timestep
 */

#include <memory>
#include <string>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "history/history.hpp"
#include "packages/packages_base.hpp"
#include "pgen/problem_in.hpp"
#include "timestepper/timestepper.hpp"

using bc::BoundaryConditions;

class Driver {
 public:
  explicit Driver(const ProblemIn* pin);

  auto execute() -> int;

 private:
  // init
  void initialize(const ProblemIn* pin);

  ProblemIn pin_;

  std::unique_ptr<PackageManager> manager_;

  // TODO(astrobarker): thread in run_id_
  // std::string run_id_;
  int nX_;
  std::string problem_name_;
  bool restart_;

  std::unique_ptr<BoundaryConditions> bcs_;

  double time_;
  double dt_;
  double t_end_;
  double cfl_;
  int i_print_;
  double nlim_;
  double dt_hdf5_;
  double dt_init_frac_;

  // core bits
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
}; // class Driver
//
namespace {

/**
 * Compute the CFL timestep restriction.
 **/
KOKKOS_INLINE_FUNCTION
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
} // namespace
