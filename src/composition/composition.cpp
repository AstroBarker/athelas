#include "composition/composition.hpp"

#include "utils/error.hpp"

auto CompositionKey::add_element(int Z, double atomic_mass, int max_charge) -> size_t {
  elements_.emplace_back(Z, atomic_mass, max_charge);
  const size_t idx = elements_.size() - 1;
  element_map_[Z] = idx;
  
  total_states_ += max_charge + 1;
  
  return idx;
}

// TODO(astrobarker): should this throw?
[[nodiscard]] auto CompositionKey::get_element_index(const int Z) const -> size_t {
 if (!element_map_.contains(Z)) {
    THROW_ATHELAS_ERROR("Element '" + std::to_string(Z) + "' not found");
 }
  return element_map_.at(Z);
}

[[nodiscard]] auto CompositionKey::get_state_offset(int element_idx) const -> int {
  int offset = 0;
  for (int i = 0; i < element_idx; ++i) {
    offset += elements_[i].n_states();
  }
  return offset;
}

[[nodiscard]] auto CompositionKey::n_species() const noexcept -> size_t { return elements_.size(); }
[[nodiscard]] auto CompositionKey::n_total_states() const noexcept -> int { return total_states_; }
[[nodiscard]] auto CompositionKey::get_element(int idx) const noexcept -> const Element& { return elements_[idx]; }

// --- CompositionData ---


CompositionData::CompositionData(const CompositionKey& comp_mgr, const int nX, const int nNodes)
  : comp_mgr_(&comp_mgr), nX_(nX), nNodes_(nNodes) {
  
  const size_t n_species = comp_mgr_->n_species();
  if (n_species > 0) {
    mass_fractions_ = View3D<double>("mass_fractions", nX_, nNodes_, n_species);
      
    // Ionization fractions: [nX][nSpecies][nNodes][charge_state]
    ionization_fractions_ = View4D<double>("ionization_fractions", nX_,
                                             n_species, nNodes_, get_max_states());

    charge_ = View1D<int>("chargev", n_species);
  }
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION
auto CompositionData::element_mass_fraction(const int i, const int node, const int element_idx) const -> double {
  return mass_fractions_(i, node, element_idx);
}

[[nodiscard]] auto CompositionData::ionization_fractions() const noexcept -> View4D<double> { return ionization_fractions_; }

// Validate mass fractions sum to 1 across elements, convenience
[[nodiscard]] KOKKOS_INLINE_FUNCTION
auto CompositionData::sum_mass_fractions(const int i, const int node) const -> double {
  double total_X = 0.0;
  const size_t n_species = comp_mgr_->n_species();

  for (int elem = 0; elem < n_species; ++elem) {
    total_X += mass_fractions_(i, node, elem);
  }
  return total_X; // Should be ~1.0
}
// Compute total element number density (all ionization states)
KOKKOS_INLINE_FUNCTION
auto element_number_density(double mass_frac, double atomic_mass, double rho) -> double {
  return (mass_frac * rho) / (atomic_mass * constants::amu_to_g);
}

// Compute electron number density (derived quantity)
KOKKOS_INLINE_FUNCTION
auto electron_density(const View3D<double> mass_fractions, const View4D<double> ion_fractions, const View1D<int> charges, int ix, int node, double rho) -> double {
  double n_e = 0.0;
  const size_t n_species = charges.size();
  
  for (int elem = 0; elem < n_species; ++elem) {
    const double ne_elem = element_number_density(mass_fractions(ix, node, elem), charges(elem), rho);
    
    // Sum charge * ionization_fraction for each charge state
    const int max_charge = charges(elem);
    for (int charge = 1; charge <= max_charge; ++charge) {
      const double f_ion = ion_fractions(ix, node, elem, charge);
      n_e += charge * f_ion * ne_elem;
    }
  }
  return n_e;
}
