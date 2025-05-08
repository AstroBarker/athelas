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

#include <iostream>
#include <string>

#include "boundary_conditions.hpp"
#include "boundary_conditions_base.hpp"
#include "grid.hpp"
#include "utilities.hpp"

namespace bc {

BcType parse_bc_type(const std::string &name) {
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
  THROW_ATHELAS_ERROR(" ! bc_type not known!");
  return BcType::Null;
}

BoundaryConditions make_boundary_conditions(
    bool do_rad,

    const std::string& fluid_bc_i,
    const std::string& fluid_bc_o,
    const std::array<Real, 3>& fluid_i_dirichlet_values,
    const std::array<Real, 3>& fluid_o_dirichlet_values,

    const std::string& rad_bc_i,
    const std::string& rad_bc_o,
    const std::array<Real, 2>& rad_i_dirichlet_values,
    const std::array<Real, 2>& rad_o_dirichlet_values
) {
    BoundaryConditions my_bc;

    // --- Fluid BCs ---
    BcType f_inner = parse_bc_type(fluid_bc_i);
    BcType f_outer = parse_bc_type(fluid_bc_o);

    my_bc.fluid_bc[0] = (f_inner == BcType::Dirichlet)
        ? BoundaryConditionsData<3>(f_inner, fluid_i_dirichlet_values)
        : BoundaryConditionsData<3>(f_inner);

    my_bc.fluid_bc[1] = (f_outer == BcType::Dirichlet)
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

        my_bc.rad_bc[0] = (r_inner == BcType::Dirichlet)
            ? BoundaryConditionsData<2>(r_inner, rad_i_dirichlet_values)
            : BoundaryConditionsData<2>(r_inner);

        my_bc.rad_bc[1] = (r_outer == BcType::Dirichlet)
            ? BoundaryConditionsData<2>(r_outer, rad_o_dirichlet_values)
            : BoundaryConditionsData<2>(r_outer);
    }

    return my_bc;
}
} // namespace bc
