#pragma once

#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"

void fill_derived_comps(State *state, const GridStructure *grid, const ModalBasis *basis);
void fill_derived_ionization(State *state, const GridStructure *grid, const ModalBasis *basis);

KOKKOS_FUNCTION
void paczynski_terms(const State *state,
                     int ix, int node, double *lambda);

// Compute total element number density (all ionization states)
KOKKOS_FUNCTION
auto element_number_density(double mass_frac, double atomic_mass, double rho)
    -> double;

// Compute electron number density (derived quantity)
KOKKOS_FUNCTION
auto electron_density(const View3D<double> mass_fractions,
                      const View4D<double> ion_fractions,
                      const View1D<int> charges, int ix, int node, double rho)
    -> double;
