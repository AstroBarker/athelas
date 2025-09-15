#pragma once

#include <memory>

#include "atom/atom.hpp"
#include "utils/abstractions.hpp"

using atom::AtomicData;

class IonizationState {
 public:
  IonizationState(int nX, int nNodes, int n_species, int n_states,
                  const std::string &fn_ionization,
                  const std::string &fn_degeneracy);

  [[nodiscard]] auto ionization_fractions() const noexcept -> View4D<double>;
  [[nodiscard]] auto atomic_data() const noexcept -> AtomicData *;

 private:
  View4D<double> ionization_fractions_; // [nX][nNodes][n_species][max_charge+1]
  std::unique_ptr<AtomicData> atomic_data_;
};

// Composition data handler - manages mass fractions and ionization fractions
class CompositionData {
 public:
  CompositionData(int nX, int order, int n_species);

  [[nodiscard]] auto mass_fractions() const noexcept -> View3D<double>;

  [[nodiscard]] auto charge() const noexcept -> View1D<int>;

  [[nodiscard]] auto n_species() const noexcept -> size_t {
    return mass_fractions_.extent(2);
  }

 private:
  int nX_, order_, n_species_;

  View3D<double> mass_fractions_; // [nX][order][n_species]
  View1D<int> charge_; // n_species
}; // class CompositionData

// Compute total element number density (all ionization states)
KOKKOS_FUNCTION
auto element_number_density(double mass_frac, double atomic_mass, double rho)
    -> double;

// Compute electron number density (derived quantity)
KOKKOS_INLINE_FUNCTION
auto electron_density(const View3D<double> mass_fractions,
                      const View4D<double> ion_fractions,
                      const View1D<int> charges, int ix, int node, double rho)
    -> double;
