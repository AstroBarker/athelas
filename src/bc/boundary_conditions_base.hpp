#pragma once
/**
 * @file boundary_conditions_base.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Boundary conditions base structures
 */

#include <array>
#include <cassert>

#include "pgen/problem_in.hpp"
#include "utils/error.hpp"

namespace bc {

enum class BcType : int {
    Outflow,
    Dirichlet,
    Reflecting,
    Periodic,
    Null // don't go here
};

BcType parse_bc_type(const std::string &name);

template <int N>
struct BoundaryConditionsData {
    BcType type;
    Real dirichlet_values[N] = {}; // Only used if type == Dirichlet
    Real time; // placeholder
    
    // necessary
    KOKKOS_INLINE_FUNCTION
    BoundaryConditionsData()
        : type(BcType::Outflow)
    {}

    KOKKOS_INLINE_FUNCTION
    BoundaryConditionsData(BcType type_)
        : type(type_)
    {}

    KOKKOS_INLINE_FUNCTION
    BoundaryConditionsData(BcType type_, const std::array<Real, N> vals)
        : type(type_)
    {
        assert(type_ == BcType::Dirichlet && 
                "This constructor is for Dirichlet boundary conditions!\n");
        for (int i = 0; i < N; ++i) {
            dirichlet_values[i] = vals[i];
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real get_dirichlet_value(int i) const {
        return dirichlet_values[i];
    }
};

constexpr static int NUM_HYDRO_VARS = 3;
constexpr static int NUM_RAD_VARS = 2;
struct BoundaryConditions {
    // in the below arrays, 0 is inner boundary, 1 is outer
    std::array<BoundaryConditionsData<NUM_HYDRO_VARS>, 2> fluid_bc;
    std::array<BoundaryConditionsData<NUM_RAD_VARS>, 2> rad_bc;
    bool do_rad = false;
};

// --- helper functions to pull out bc ---
template <int N>
KOKKOS_INLINE_FUNCTION
std::array<BoundaryConditionsData<N>, 2> get_bc_data(BoundaryConditions* bc);

template <>
KOKKOS_INLINE_FUNCTION
std::array<BoundaryConditionsData<3>, 2> get_bc_data<3>(BoundaryConditions* bc) {
    return bc->fluid_bc;
}

template <>
KOKKOS_INLINE_FUNCTION
std::array<BoundaryConditionsData<2>, 2> get_bc_data<2>(BoundaryConditions* bc) {
    assert(bc->do_rad && "Need radiation enabled to get radiation bcs!\n");
    return bc->rad_bc;
}

// Applies boundary condition for one variable `q`
template <int N>
KOKKOS_INLINE_FUNCTION
void apply_bc(
    const BoundaryConditionsData<N>& bc,
    View3D<Real> U,
    const int q,
    const int ghost_cell,
    const int interior_cell,
    const int num_modes
) {
    switch (bc.type) {
        case BcType::Outflow:
            for (int k = 0; k < num_modes; k++) {
              U(q, ghost_cell, k) = U(q, interior_cell, k);
            }
            break;

        // NOTE: Literally the same as the above, but the required 
        // use is different. interior_cell should be of the opposite
        // side as ghost_cell. Not ideal, but works.
        case BcType::Periodic:
            assert(interior_cell != ghost_cell + 1 && "Bad use of periodic BC!\n");
            assert(interior_cell != ghost_cell - 1 && "Bad use of periodic BC!\n");
            for (int k = 0; k < num_modes; k++) {
              U(q, ghost_cell, k) = U(q, interior_cell, k);
            }
            break;

        case BcType::Reflecting:
            for (int k = 0; k < num_modes; k++) {
                if (q == 1) {  // Momentum (q == 1)
                    // Reflect momentum in the cell average (k == 0) and leave higher modes unchanged (k > 0)
                    U(q, ghost_cell, k) = (k == 0) ? -U(q, interior_cell, k) : U(q, interior_cell, k);
                } else {  // Non-momentum variables
                    // Reflect cell averages (k == 0) and invert higher modes (k > 0)
                    U(q, ghost_cell, k) = (k == 0) ? U(q, interior_cell, k) : -U(q, interior_cell, k);
                }
            }
            break;

        // TODO(astrobarker): could need extending
        case BcType::Dirichlet:
              U(q, ghost_cell, 0) = bc.dirichlet_values[q];
            for (int k = 1; k < num_modes; k++) {
              U(q, ghost_cell, k) = 0.0; // slopes++ set to 0
            }
            break;

        case BcType::Null:
            THROW_ATHELAS_ERROR("Null BC is not for use!");
            break;
    }
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
);

} // namespace bc
