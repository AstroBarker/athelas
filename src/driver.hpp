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
#include "eos/eos.hpp"
#include "geometry/grid.hpp"
#include "opacity/opac.hpp"
#include "pgen/problem_in.hpp"
#include "timestepper/timestepper.hpp"
#include "utils/abstractions.hpp"
#include "utils/error.hpp"

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

  Real time_;
  Real dt_;
  Real t_end_;
  Real cfl_;
  int i_print_;
  Real nlim_;
  Real dt_hdf5_;
  Real dt_init_frac_;

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

  std::unique_ptr<ModalBasis> basis_; // init in constr body
}; // class Driver
