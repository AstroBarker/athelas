/**
 * @file timestepper.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Primary time marching routine
 */

#include <vector>

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "timestepper/tableau.hpp"
#include "timestepper/timestepper.hpp"

/**
 * The constructor creates the necessary data structures for time evolution.
 * Lots of structures used in discretizations live here.
 **/
TimeStepper::TimeStepper(const ProblemIn* pin, GridStructure* grid, EOS* eos)
    : mSize_(grid->get_n_elements() + 2),
      integrator_(
          create_tableau(pin->param()->get<MethodID>("time.integrator"))),
      nStages_(integrator_.num_stages), tOrder_(integrator_.explicit_order),
      U_s_("U_s", nStages_ + 1,
           3 + (pin->param()->get<bool>("physics.rad_active")) * 2, mSize_ + 1,
           pin->param()->get<int>("fluid.porder")),
      dU_s_("dU_s", nStages_ + 1,
            3 + (pin->param()->get<bool>("physics.rad_active")) * 2,
            mSize_ + 1, pin->param()->get<int>("fluid.porder")),
      dU_s_implicit_("dU_s_implicit", nStages_ + 1, 5, mSize_ + 1,
                     pin->param()->get<int>("fluid.porder")),
      SumVar_U_("SumVar_U",
                3 + (pin->param()->get<bool>("physics.rad_active")) * 2,
                mSize_ + 1, pin->param()->get<int>("fluid.porder")),
      grid_s_(nStages_ + 1, GridStructure(pin)),
      stage_data_("StageData", nStages_ + 1, mSize_ + 1),
      flux_u_("flux_u_", nStages_ + 1, mSize_ + 1), eos_(eos) {}

[[nodiscard]] auto TimeStepper::get_n_stages() const noexcept -> int {
  return integrator_.num_stages;
}
