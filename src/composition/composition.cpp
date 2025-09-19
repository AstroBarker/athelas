#include "composition/composition.hpp"
#include "utils/constants.hpp"

/**
 * @brief Fill derived composition quantities
 *
 * Currently, fills number densities.
 *
 * TODO(astrobarker): Explore hierarchical parallelism for inner loops
 */
void fill_derived_comps(State *const state, const GridStructure *const grid,
                        const ModalBasis *const basis) {
  static constexpr int ilo = 1;
  static const auto &ihi = grid->get_ihi();
  static const auto &nnodes = grid->get_n_nodes();

  auto *const comps = state->comps();
  const auto mass_fractions = comps->mass_fractions();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  auto number_density = comps->number_density();
  const size_t num_species = comps->n_species();

  static constexpr double inv_m_p = 1.0 / constants::m_p;
  Kokkos::parallel_for(
      "Composition :: fill derived",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, nnodes + 2}),
      KOKKOS_LAMBDA(const int ix, const int node) {
        double n = 0.0;
        for (size_t e = 0; e < num_species; ++e) {
          const double A = species(e) + neutron_number(e);
          const double xk = basis->basis_eval(mass_fractions, ix, e, node);
          n += xk / A;
        }
        number_density(ix, node) = n * inv_m_p;
      });
}

/**
 * @brief Fill derived ionization quantities
 *
 * These are quantities needed for the Paczynski eos.
 *
 * TODO(astrobarker): Explore hierarchical parallelism for inner loops
 */
void fill_derived_ionization(State *const state,
                             const GridStructure *const grid,
                             const ModalBasis *const basis) {
  static constexpr int ilo = 1;
  static const auto &ihi = grid->get_ihi();
  static const auto &nnodes = grid->get_n_nodes();

  const auto *const comps = state->comps();
  const auto mass_fractions = comps->mass_fractions();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  const auto number_density = comps->number_density();
  const size_t num_species = comps->n_species();

  auto *const ionization_states = state->ionization_state();
  const auto ionization_fractions = ionization_states->ionization_fractions();
  auto ybar = ionization_states->ybar();
  auto e_ion_corr = ionization_states->e_ion_corr();
  auto sigma1 = ionization_states->sigma1();
  auto sigma2 = ionization_states->sigma2();
  auto sigma3 = ionization_states->sigma3();

  // pull out atomic data containers
  const auto *const atomic_data = ionization_states->atomic_data();
  const auto ion_data = atomic_data->ion_data();
  const auto species_offsets = atomic_data->offsets();

  const auto ucf = state->u_cf();

  // NOTE: check index ranges inside here when saha ncomps =/= num_species
  Kokkos::parallel_for(
      "Ionization :: fill derived",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, nnodes + 2}),
      KOKKOS_LAMBDA(const int ix, const int node) {
        const double rho = 1.0 / basis->basis_eval(ucf, ix, 0, node);
        // This kernel is horrible.
        // Reduce the ionization based quantities sigma1-3, e_ion_corr
        for (size_t e = 0; e < num_species; ++e) {
          // pull out element info
          const auto species_atomic_data =
              species_data(ion_data, species_offsets, e);
          const auto ionization_fractions_e =
              Kokkos::subview(ionization_fractions, ix, node, e, Kokkos::ALL);
          const size_t nstates = e + 1;

          // 1. Get lmax -- index associated with max ionization per species
          size_t lmax = 0;
          double ymax = 0;
          for (size_t i = 0; i < nstates; ++i) {
            const double y = ionization_fractions_e(i);
            if (y > ymax) {
              ymax = y;
              lmax = i;
            }
          }

          // 2. Sum ionization fractions * ionization potentials for e_ion_corr
          double sum_ion_pot = 0.0;
          for (size_t i = 0; i < nstates; ++i) {
            // I think that this pattern is not optimal.
            double sum_pot = 0.0;
            for (size_t m = 0; m < i; ++m) {
              sum_pot += species_atomic_data(i).chi;
            }
            sum_ion_pot += ionization_fractions_e(i) * sum_pot;
          }

          // 3. Find two most populated states and store the higher as y_r.
          // chi_r is the ionization potential between these states.
          // Check index logic.
          // Wish I could avoid branching logic...
          double y_r = 0;
          double chi_r = 0.0;
          if (lmax == 0) {
            y_r = ionization_fractions_e(lmax);
            chi_r = species_atomic_data(lmax).chi;
          } else if (lmax == (e + 0)) {
            y_r = ionization_fractions_e(lmax);
            chi_r = species_atomic_data(lmax - 1).chi;
          } else {
            // Comparison between lmax+1 and lmax-1 indices
            if (ionization_fractions_e(lmax + 1) >
                ionization_fractions_e(lmax - 1)) {
              y_r = ionization_fractions_e(lmax + 1);
              chi_r = species_atomic_data(lmax).chi;
            } else {
              y_r = ionization_fractions_e(lmax);
              chi_r = species_atomic_data(lmax - 1).chi;
            }
          }

          // 4. The good stuff -- integrate the various sigma terms
          // and the internal energy term from partial ionization.
          // Start with constructing the abundance n_k

          const double atomic_mass = species(e) + neutron_number(e);
          const double xk = basis->basis_eval(mass_fractions, ix, e, node);
          const double nk = element_number_density(xk, atomic_mass, rho);
          sigma1(ix, node) += nk * y_r * (1 - y_r); // sigma1
          sigma2(ix, node) += chi_r * sigma1(ix, node); // sigma2
          sigma3(ix, node) += chi_r * sigma2(ix, node); // sigma3
          e_ion_corr(ix, node) +=
              number_density(ix, node) * nk * sum_ion_pot; // e_ion_corr
        }
      });
}

/**
 * @brief Store the extra "lambda" terms for paczynski eos
 * NOTE:: Lambda contents:
 * 0: N (for ion pressure)
 * 1: ye
 * 2: ybar (mean ionization state)
 * 3: sigma1
 * 4: sigma2
 * 5: sigma3
 * 6: e_ioncorr (ionization corrcetion to internal energy)
 * 7: temperature_guess
 *
 * TODO(astrobarker): should inputs to this be subviews?
 */
KOKKOS_FUNCTION
void paczynski_terms(const State *const state, const int ix, const int node,
                     double *const lambda) {
  const auto ucf = state->u_cf();
  const auto uaf = state->u_af();

  const auto *const comps = state->comps();
  const auto mass_fractions = comps->mass_fractions();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  const auto number_density = comps->number_density();
  const auto ye = comps->ye();

  const auto *const ionization_states = state->ionization_state();
  const auto ionization_fractions = ionization_states->ionization_fractions();
  const auto ybar = ionization_states->ybar();
  const auto e_ion_corr = ionization_states->e_ion_corr();
  const auto sigma1 = ionization_states->sigma1();
  const auto sigma2 = ionization_states->sigma2();
  const auto sigma3 = ionization_states->sigma3();

  lambda[0] = number_density(ix, node);
  lambda[1] = ye(ix, node);
  lambda[2] = ybar(ix, node);
  lambda[3] = sigma1(ix, node);
  lambda[4] = sigma2(ix, node);
  lambda[5] = sigma3(ix, node);
  lambda[6] = e_ion_corr(ix, node);
  lambda[7] = uaf(ix, node, 1); // temperature
}

// Compute total element number density
KOKKOS_FUNCTION
auto element_number_density(const double mass_frac, const double atomic_mass,
                            const double rho) -> double {
  return (mass_frac * rho) / (atomic_mass * constants::amu_to_g);
}

// Compute electron number density
KOKKOS_FUNCTION
auto electron_density(const View3D<double> mass_fractions,
                      const View4D<double> ion_fractions,
                      const View1D<int> charges, int ix, int node, double rho)
    -> double {
  double n_e = 0.0;
  const size_t n_species = charges.size();

  Kokkos::parallel_reduce(
      "Paczynski::Reduce::ne", n_species,
      KOKKOS_LAMBDA(const int elem, double &ne_local) {
        const double n_elem = element_number_density(
            mass_fractions(ix, node, elem), charges(elem), rho);

        // Sum charge * ionization_fraction for each charge state
        const int max_charge = charges(elem);
        for (int charge = 1; charge <= max_charge; ++charge) {
          const double f_ion = ion_fractions(ix, node, elem, charge);
          ne_local += charge * f_ion * n_elem;
        }
      },
      Kokkos::Sum<double>(n_e));
  return n_e;
}
