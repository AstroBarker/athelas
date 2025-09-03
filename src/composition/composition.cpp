#include "composition/composition.hpp"

#include "atom/atom.hpp"
#include "utils/error.hpp"

CompositionData::CompositionData(const int nX, const int nNodes,
                                 const int n_species)
    : nX_(nX), nNodes_(nNodes), n_species_(n_species) {

  if (n_species <= 0) {
    THROW_ATHELAS_ERROR("CompositionData :: n_species must be > 0!");
  }
  mass_fractions_ = View3D<double>("mass_fractions", nX_, nNodes_, n_species);
  charge_ = View1D<int>("charge", n_species);
}

[[nodiscard]] auto CompositionData::mass_fractions() const noexcept
    -> View3D<double> {
  return mass_fractions_;
}
[[nodiscard]] auto CompositionData::charge() const noexcept -> View1D<int> {
  return charge_;
}

// Validate mass fractions sum to 1 across elements, convenience
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto
CompositionData::sum_mass_fractions(const int i, const int node) const
    -> double {
  double total_X = 0.0;

  for (int elem = 0; elem < n_species_; ++elem) {
    total_X += mass_fractions_(i, node, elem);
  }
  return total_X; // Should be ~1.0
}

// --- end CompositionData ---

// NOTE: Something awful here: need an estimate of the max number of ionization
// states
//  when constructing the comps object
//  We allocate in general more data than necessary
IonizationState::IonizationState(const int nX, const int nNodes,
                                 const int n_species, const int n_states,
                                 const std::string& fn_ionization,
                                 const std::string& fn_degeneracy)
    : ionization_fractions_("ionization_fractions", nX, nNodes, n_species,
                            n_states),
      atomic_data_(fn_ionization, fn_degeneracy) {
  if (n_species <= 0) {
    THROW_ATHELAS_ERROR("IonizationState :: n_species must be > 0!");
  }
  if (n_states <= 0) {
    THROW_ATHELAS_ERROR("IonizationState :: n_states must be > 0!");
  }
}

[[nodiscard]] auto IonizationState::ionization_fractions() const noexcept
    -> View4D<double> {
  return ionization_fractions_;
}

[[nodiscard]] auto IonizationState::atomic_data() const noexcept -> AtomicData {
  return atomic_data_;
}

// Compute total element number density (all ionization states)
KOKKOS_INLINE_FUNCTION
auto element_number_density(double mass_frac, double atomic_mass, double rho)
    -> double {
  return (mass_frac * rho) / (atomic_mass * constants::amu_to_g);
}

// Compute electron number density (derived quantity)
KOKKOS_INLINE_FUNCTION
auto electron_density(const View3D<double> mass_fractions,
                      const View4D<double> ion_fractions,
                      const View1D<int> charges, int ix, int node, double rho)
    -> double {
  double n_e = 0.0;
  const size_t n_species = charges.size();

  for (size_t elem = 0; elem < n_species; ++elem) {
    const double ne_elem = element_number_density(
        mass_fractions(ix, node, elem), charges(elem), rho);

    // Sum charge * ionization_fraction for each charge state
    const int max_charge = charges(elem);
    for (int charge = 1; charge <= max_charge; ++charge) {
      const double f_ion = ion_fractions(ix, node, elem, charge);
      n_e += charge * f_ion * ne_elem;
    }
  }
  return n_e;
}
