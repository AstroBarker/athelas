#include "composition/composition.hpp"

#include "atom/atom.hpp"
#include "utils/error.hpp"
#include <memory>

CompositionData::CompositionData(const int nX, const int order,
                                 const int n_species)
    : nX_(nX), order_(order), n_species_(n_species) {

  if (n_species <= 0) {
    THROW_ATHELAS_ERROR("CompositionData :: n_species must be > 0!");
  }
  mass_fractions_ = View3D<double>("mass_fractions", nX_, order, n_species);
  charge_ = View1D<int>("charge", n_species);
}

[[nodiscard]] auto CompositionData::mass_fractions() const noexcept
    -> View3D<double> {
  return mass_fractions_;
}
[[nodiscard]] auto CompositionData::charge() const noexcept -> View1D<int> {
  return charge_;
}

// --- end CompositionData ---

// NOTE: Something awful here: need an estimate of the max number of ionization
// states
//  when constructing the comps object
//  We allocate in general more data than necessary
//
//  TODO(astrobarker): flatten last two dimensions to reduce memory footprint.
//  See atom.hpp
IonizationState::IonizationState(const int nX, const int nNodes,
                                 const int n_species, const int n_states,
                                 const std::string& fn_ionization,
                                 const std::string& fn_degeneracy)
    : ionization_fractions_("ionization_fractions", nX, nNodes + 2, n_species,
                            n_states),
      atomic_data_(std::make_unique<AtomicData>(fn_ionization, fn_degeneracy)) {
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

[[nodiscard]] auto IonizationState::atomic_data() const noexcept
    -> AtomicData* {
  return atomic_data_.get();
}

// Compute total element number density (all ionization states)
KOKKOS_FUNCTION
auto element_number_density(const double mass_frac, const double atomic_mass,
                            const double rho) -> double {
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
