/**
 * @file boundary_conditions.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Boundary conditions
 *
 * @details Implemented BCs
 *            - outflow
 *            - reflecting
 *            - periodic
 *            - Dirichlet
 */

#include <string>

#include "boundary_conditions_base.hpp"
#include "utils/error.hpp"

namespace bc {

BcType parse_bc_type(const std::string& name) {
  if (name == "outflow") {
    return BcType::Outflow;
  }
  if (name == "reflecting") {
    return BcType::Reflecting;
  }
  if (name == "periodic") {
    return BcType::Periodic;
  }
  if (name == "dirichlet") {
    return BcType::Dirichlet;
  }
  if (name == "marshak") {
    return BcType::Marshak;
  }
  THROW_ATHELAS_ERROR(" ! bc_type not known!");
  return BcType::Null;
}

auto make_boundary_conditions(const ProblemIn* pin) -> BoundaryConditions {
  BoundaryConditions my_bc;
  const auto do_rad     = pin->param()->get<bool>("physics.rad_active");
  const auto fluid_bc_i = pin->param()->get<std::string>("fluid.bc.i");
  const auto fluid_bc_o = pin->param()->get<std::string>("fluid.bc.o");
  const auto fluid_i_dirichlet_values =
      pin->param()->get<std::array<double, 3>>("fluid.bc.i.dirichlet_values");
  const auto fluid_o_dirichlet_values =
      pin->param()->get<std::array<double, 3>>("fluid.bc.o.dirichlet_values");
  const auto rad_bc_i = pin->param()->get<std::string>("radiation.bc.i", "outflow");
  const auto rad_bc_o = pin->param()->get<std::string>("radiation.bc.o", "outflow");
  const auto rad_i_dirichlet_values =
      pin->param()->get<std::array<double, 2>>("radiation.bc.i.dirichlet_values", {0.0, 0.0});
  const auto rad_o_dirichlet_values =
      pin->param()->get<std::array<double, 2>>("radiation.bc.o.dirichlet_values", {0.0, 0.0});

  // --- Fluid BCs ---
  BcType f_inner = parse_bc_type(fluid_bc_i);
  BcType f_outer = parse_bc_type(fluid_bc_o);

  my_bc.fluid_bc[0] =
      (f_inner == BcType::Dirichlet)
          ? BoundaryConditionsData<3>(f_inner, fluid_i_dirichlet_values)
          : BoundaryConditionsData<3>(f_inner);

  my_bc.fluid_bc[1] =
      (f_outer == BcType::Dirichlet)
          ? BoundaryConditionsData<3>(f_outer, fluid_o_dirichlet_values)
          : BoundaryConditionsData<3>(f_outer);

  // --- Radiation BCs ---
  if (do_rad) {
    if (rad_bc_i == "" || rad_bc_o == "") {
      THROW_ATHELAS_ERROR(" ! Radiation enabled but rad_bc_i/o is not set.");
    }

    my_bc.do_rad = true;

    BcType r_inner = parse_bc_type(rad_bc_i);
    BcType r_outer = parse_bc_type(rad_bc_o);

    my_bc.rad_bc[0] =
        (r_inner == BcType::Dirichlet || r_inner == BcType::Marshak)
            ? BoundaryConditionsData<2>(r_inner, rad_i_dirichlet_values)
            : BoundaryConditionsData<2>(r_inner);

    my_bc.rad_bc[1] =
        (r_outer == BcType::Dirichlet || r_outer == BcType::Marshak)
            ? BoundaryConditionsData<2>(r_outer, rad_o_dirichlet_values)
            : BoundaryConditionsData<2>(r_outer);
  }

  return my_bc;
}
} // namespace bc
