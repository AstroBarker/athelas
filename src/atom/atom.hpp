#pragma once
/*
 * @file atom.hpp
 * --------------
 *
 * @brief Atomic data structures
 * @note Used to hold NIST atomic data
 *   Designed for an array of structures pattern,
 *   where algorithms should loop over species, then cells.
 */

#include <vector>

#include "utils/abstractions.hpp"

namespace atom {

struct EnergyLevel {
  double ionization_potential; // in eV
  double statistical_weight; // degeneracy factor
};

struct AtomData {
  View1D<EnergyLevel> levels;
  View1D<size_t> level_offsets;
  View1D<size_t> num_levels;
};

using ViewLevel     = Kokkos::View<EnergyLevel*>;
using ViewSize      = Kokkos::View<size_t*>;
using ViewLevelHost = ViewLevel::HostMirror;
using ViewSizeHost  = ViewSize::HostMirror;

int num_species = 5;
std::vector<std::vector<EnergyLevel>> host_levels(num_species);
// Fill host_levels[s] with per-species data

// Compute total size and offsets
size_t total_levels = 0;
std::vector<size_t> offsets(num_species);
for (int s = 0; s < num_species; ++s) {
  offsets[s] = total_levels;
  total_levels += host_levels[s].size();
}

// Allocate views
ViewLevel levels("energy_levels", total_levels);
ViewSize level_offsets("level_offsets", num_species);
ViewSize level_counts("level_counts", num_species);

ViewLevelHost h_levels = Kokkos::create_mirror_view(levels);
ViewSizeHost h_offsets = Kokkos::create_mirror_view(level_offsets);
ViewSizeHost h_counts  = Kokkos::create_mirror_view(level_counts);

// Fill
for (int s = 0; s < num_species; ++s) {
  h_offsets(s) = offsets[s];
  h_counts(s)  = host_levels[s].size();
  for (size_t l = 0; l < host_levels[s].size(); ++l) {
    h_levels(offsets[s] + l) = host_levels[s][l];
  }
}

// Deep copy to device
Kokkos::deep_copy(levels, h_levels);
Kokkos::deep_copy(level_offsets, h_offsets);
Kokkos::deep_copy(level_counts, h_counts);

KOKKOS_INLINE_FUNCTION
double partition_function(int species_idx, double T_eV,
                          Kokkos::View<const EnergyLevel*> levels,
                          Kokkos::View<const size_t*> offsets,
                          Kokkos::View<const size_t*> counts) {
  size_t start = offsets(species_idx);
  size_t count = counts(species_idx) // num_levels

      double Z = 0.0;
  for (size_t l = 0; l < count; ++l) {
    const auto& level = levels(start + l);
    Z += level.statistical_weight *
         Kokkos::exp(-level.ionization_potential / T_eV);
  }
  return Z;
}
Kokkos::parallel_for(
    "compute_partition", N_cells, KOKKOS_LAMBDA(int i) {
      for (int s = 0; s < num_species; ++s) {
        double Z = partition_function(s, temperature(i), levels, level_offsets,
                                      level_counts);
        // use or store Z
      }
    });

} // namespace atom
