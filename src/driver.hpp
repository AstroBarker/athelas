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
//#include "bc/boundary_conditions_base.hpp"
#include "eos/eos.hpp"
#include "pgen/problem_in.hpp"
#include "geometry/grid.hpp"
#include "opacity/opac.hpp"
#include "timestepper/timestepper.hpp"
#include "utils/abstractions.hpp"
#include "utils/error.hpp"

using bc::BoundaryConditions;

class Driver {
 public:
  explicit Driver( const ProblemIn* pin );

  auto execute( ) -> int;

 private:
  // init
  void initialize( const ProblemIn* pin );

  ProblemIn pin_;

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
  EOS eos_;
  Opacity opac_;
  GridStructure grid_;
  Options opts_;
  State state_;

  // slope limiters
  SlopeLimiter sl_hydro_;
  SlopeLimiter sl_rad_;

  // timestepper
  TimeStepper ssprk_;

  std::unique_ptr<ModalBasis> fluid_basis_; // init in constr body
  std::unique_ptr<ModalBasis> radiation_basis_; // init in constr body
}; // class Driver
