#pragma once

#include <memory>

#include "atom/atom.hpp"
#include "utils/abstractions.hpp"

namespace athelas::atom {

/**
 * @class IonizationState
 * @brief class for holding ionization state. We store here the ionization
 * fractions of species, an AtomicData object (see atom.hpp), the mean
 * ionizaiton state ybar, and several quantities needed in the
 * Paczynski eos for ionization corrections.
 */
class IonizationState {
 public:
  IonizationState(int nX, int nNodes, int n_species, int n_states,
                  const std::string &fn_ionization,
                  const std::string &fn_degeneracy);

  [[nodiscard]] auto ionization_fractions() const noexcept -> View4D<double>;
  [[nodiscard]] auto atomic_data() const noexcept -> AtomicData *;
  [[nodiscard]] auto ybar() const noexcept -> View2D<double>;
  [[nodiscard]] auto e_ion_corr() const noexcept -> View2D<double>;
  [[nodiscard]] auto sigma1() const noexcept -> View2D<double>;
  [[nodiscard]] auto sigma2() const noexcept -> View2D<double>;
  [[nodiscard]] auto sigma3() const noexcept -> View2D<double>;

 private:
  View4D<double> ionization_fractions_; // [nX][nNodes][n_species][max_charge+1]
  std::unique_ptr<AtomicData> atomic_data_;

  // Derived quantities for Paczynski, stored nodally
  View2D<double> ybar_; // mean ionization fraction
  View2D<double> e_ion_corr_; // ionization correction to internal energy
  View2D<double> sigma1_;
  View2D<double> sigma2_;
  View2D<double> sigma3_;
};

/**
 * @class CompositionData
 * TODO(astrobarker): probably moving mass fractions into ucf soon.
 */
class CompositionData {
 public:
  CompositionData(int nX, int order, int n_species);

  [[nodiscard]] auto mass_fractions() const noexcept -> View3D<double>;

  [[nodiscard]] auto charge() const noexcept -> View1D<int>;

  [[nodiscard]] auto neutron_number() const noexcept -> View1D<int>;

  [[nodiscard]] auto ye() const noexcept -> View2D<double>;

  [[nodiscard]] auto number_density() const noexcept -> View2D<double>;

  [[nodiscard]] auto n_species() const noexcept -> size_t {
    return mass_fractions_.extent(2);
  }

 private:
  int nX_, order_, n_species_;

  View3D<double> mass_fractions_; // [nX][order][n_species]
  View2D<double> number_density_; // [nX][order] number per unit mass
  View2D<double> ye_; // [nx][nnodes]
  View1D<int> charge_; // n_species
  View1D<int> neutron_number_;
}; // class CompositionData

} // namespace athelas::atom
