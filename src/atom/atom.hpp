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

#include "io/parser.hpp"
#include "utils/abstractions.hpp"

namespace atom {

/**
 * @class AtomicData
 * @brief Class for holding atomic data -- ionization potentials and
 *   degeneracy factors for many species.
 * As it stands, the constructor assumes that atomic species are
 * contiguous. If that assumption is broken then this will be as well.
 */
class AtomicData {
 public:
  /**
   * @struct IonLevel
   * @brief Holds ionization energy and degeneracy weights for ionization level
   * Note sure if this layout is optimal or ideal. It will have some
   * data duplication.
   * Chi is the ionization potential to go form ionization state n -> n+1
   * g_lower and g_upper are the degeneracy weights of states n, n+1.
   */
  struct IonLevel {
    double chi; // ionization potential
    double g_lower; // degeneracy of lower state (n-1)
    double g_upper; // degeneracy of upper state (n)
  };

 private:
  View1D<IonLevel> ion_data_;
  View1D<int> offsets_;
  View1D<int> atomic_numbers_;
  size_t num_species_;

 public:
  AtomicData(const std::string& fn_ionization,
             const std::string& fn_degeneracy) {

    // --- load atomic data from file ---
    auto ionization_data = Parser::parse_file(fn_ionization, ' ');
    auto degeneracy_data = Parser::parse_file(fn_degeneracy, ' ');

    // -- extract columns ---
    auto [atomic_numbers, ion_charges, ionization_energies] =
        get_columns_by_indices<int, int, double>(*ionization_data, 0, 1, 2);
    auto [atomic_numbers_2, ion_charges_, weights] =
        get_columns_by_indices<int, int, double>(*degeneracy_data, 0, 1, 2);

    // NOTE: This explicitly assumes that the data in the files are ordered
    // and contiguous. If this assumption is broken, then this is incorrect.
    // TODO(astrobarker): Sorting the data after extracting would harden it.
    num_species_ = atomic_numbers.back();

    int total_levels = 0;
    std::vector<int> offs(num_species_);

    for (size_t s = 0; s < num_species_; ++s) {
      int Z   = s;
      offs[s] = total_levels;
      total_levels += Z; // Z ionization levels
    }

    ion_data_ = View1D<IonLevel>("ion_data", total_levels + num_species_ + 2);
    atomic_numbers_ = View1D<int>("atomic_number", num_species_);
    offsets_        = View1D<int>("offsets", num_species_);

    auto offs_host           = Kokkos::create_mirror_view(offsets_);
    auto ion_data_host       = Kokkos::create_mirror_view(ion_data_);
    auto atomic_numbers_host = Kokkos::create_mirror_view(atomic_numbers_);
    for (size_t s = 0; s < num_species_; ++s) {
      offs_host(s) = offs[s];
    }

    // load atomic numbers. kind of horrible
    int ind = 0;
    for (auto& z : atomic_numbers) {
      if (ind != 0) {
        if (atomic_numbers_host(ind - 1) == z) { // deal with repeats
          continue;
        }
      }
      atomic_numbers_host(ind) = z;
      ind++;
    }

    // process ionization and degerneracy data
    int chi_idx   = 0; // Index into ionization potentials vector
    int g_idx     = 0; // Index into degeneracy factors vector
    int level_idx = 0; // Index into our ion_data_ array

    for (size_t s = 0; s < num_species_; ++s) {
      int Z = atomic_numbers_host(s);

      // For species s, we have:
      // - Z ionization potentials: chi[0] to chi[Z-1]
      // - Z+1 degeneracy factors: g[0] to g[Z]
      // - Z ionization levels in our data structure

      for (int n = 0; n < Z; ++n) { // n goes from 0 to Z-1
        // Level n represents ionization from state n to state n+1
        ion_data_host(level_idx).chi     = ionization_energies[chi_idx + n];
        ion_data_host(level_idx).g_lower = weights[g_idx + n]; // g[n]
        ion_data_host(level_idx).g_upper = weights[g_idx + n + 1]; // g[n+1]

        level_idx++;
      }

      // Advance indices for next species
      chi_idx += Z;
      g_idx += Z + 1;
    }

    Kokkos::deep_copy(offsets_, offs_host);
    Kokkos::deep_copy(ion_data_, ion_data_host);
    Kokkos::deep_copy(atomic_numbers_, atomic_numbers_host);
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION auto species_data(int species) const {
    const size_t offset      = offsets_(species);
    const size_t next_offset = (species + 1 < num_species_)
                                   ? offsets_(species + 1)
                                   : ion_data_.extent(0);
    return Kokkos::subview(ion_data_, Kokkos::make_pair(offset, next_offset));
  }
};

} // namespace atom
