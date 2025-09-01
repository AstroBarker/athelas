#pragma once

#include <unordered_map>
#include <vector>

#include "utils/abstractions.hpp"
#include "utils/constants.hpp"

// Element descriptor - stores basic atomic data
struct Element {
  int Z; // atomic number
  double atomic_mass; // atomic mass in amu
  int max_charge; // maximum ionization state to track

  constexpr Element(int atomic_num, double mass, int max_ion) noexcept
      : Z(atomic_num), atomic_mass(mass), max_charge(max_ion) {}

  [[nodiscard]] auto n_states() const noexcept -> int { return max_charge + 1; }
};

// Composition manager - handles element definitions and indexing
class CompositionKey {
 public:
  CompositionKey() = default;

  // Add element (returns index for later reference)
  auto add_element(int Z, double atomic_mass, int max_charge) -> size_t;

  // Accessors
  [[nodiscard]] auto n_species() const noexcept -> size_t;
  [[nodiscard]] auto n_total_states() const noexcept -> int;
  [[nodiscard]] auto get_element(int idx) const noexcept -> const Element&;

  [[nodiscard]] auto get_element_index(int Z) const -> size_t;

  // Get state offset for flattened indexing
  [[nodiscard]] auto get_state_offset(int element_idx) const -> int;

 private:
  std::vector<Element> elements_;
  std::unordered_map<size_t, size_t> element_map_; // Z -> element index
  int total_states_ = 0;
};

// Composition data handler - manages mass fractions and ionization fractions
class CompositionData {
 public:
  CompositionData() = default;

  CompositionData(int nX, int nNodes, int n_species, int n_states);

  // Mass fraction accessors (per element)
  [[nodiscard]] auto mass_fractions() const noexcept -> View3D<double> {
    return mass_fractions_;
  }

  // Ionization fraction accessors
  [[nodiscard]] auto ionization_fractions() const noexcept -> View4D<double>;

  [[nodiscard]] auto charge() const noexcept -> View1D<int> { return charge_; }

  // Validate ionization fractions sum to 1 for each element
  [[nodiscard]] KOKKOS_INLINE_FUNCTION auto
  check_ionization_conservation(int i, int node, int element_idx) const
      -> double {
    double total_f = 0.0;
    const auto& element = charge_(element_idx);

    for (int charge = 0; charge <= element + 1; ++charge) {
      total_f += ionization_fractions_(i, node, element_idx, charge);
    }
    return total_f; // Should be ~1.0
  }

  // Validate mass fractions sum to 1 across elements, convenience
  [[nodiscard]] KOKKOS_INLINE_FUNCTION auto sum_mass_fractions(int i,
                                                               int node) const
      -> double;

  [[nodiscard]] auto n_species() const noexcept -> size_t {
    return mass_fractions_.extent(2);
  }

 private:
  int nX_, nNodes_, n_species_, n_states_;

  View3D<double> mass_fractions_; // [nX][nNodes][n_species]
  View4D<double> ionization_fractions_; // [nX][nNodes][n_species][max_charge+1]
  View1D<int> charge_; // n_species
}; // class CompositionData

// Compute total element number density (all ionization states)
KOKKOS_INLINE_FUNCTION
auto element_number_density(double mass_frac, double atomic_mass, double rho)
    -> double;

// Compute electron number density (derived quantity)
KOKKOS_INLINE_FUNCTION
auto electron_density(const View3D<double> mass_fractions,
                      const View4D<double> ion_fractions,
                      const View1D<int> charges, int ix, int node, double rho)
    -> double;
