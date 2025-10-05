#pragma once

#include "atom/atom.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "polynomial_basis.hpp"
#include "state/state.hpp"

namespace athelas::atom {

KOKKOS_FUNCTION
void solve_saha_ionization(State &state, const GridStructure &grid,
                           const eos::EOS &eos,
                           const basis::ModalBasis &fluid_basis);
KOKKOS_FUNCTION
auto saha_f(double T, const IonLevel &ion_data) -> double;
KOKKOS_FUNCTION
auto ion_frac0(double Zbar, double temperature,
               const View1D<const IonLevel> ion_datas, double nh, int min_state,
               int max_state) -> double;
KOKKOS_FUNCTION
auto saha_target(double Zbar, double T, const View1D<const IonLevel> ion_datas,
                 double nh, int min_state, int max_state) -> double;
KOKKOS_FUNCTION
auto saha_d_target(double Zbar, double T,
                   const View1D<const IonLevel> ion_datas, double nh,
                   int min_state, int max_state) -> double;

KOKKOS_FUNCTION
void saha_solve(View1D<double> ionization_states, int Z, double temperature,
                const View1D<const IonLevel> ion_datas, double nk);

} // namespace athelas::atom
