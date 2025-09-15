#include <cmath>
#include <limits>

#include "atom/atom.hpp"
#include "basis/polynomial_basis.hpp"
#include "composition/composition.hpp"
#include "composition/saha.hpp"
#include "geometry/grid.hpp"
#include "solvers/root_finder_opts.hpp"
#include "solvers/root_finders.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"
#include "utils/constants.hpp"

/**
 * @brief Functionality for saha ionization
 *
 * Word of warning: the code here is a gold medalist in index gymnastics.
 */

using atom::IonLevel;

KOKKOS_FUNCTION
void solve_saha_ionization(State &state, const GridStructure &grid,
                           const EOS &eos, const ModalBasis &fluid_basis) {
  const auto uCF = state.u_cf();
  const auto *const comps = state.comps();
  auto *const ionization_states = state.ionization_state();
  const auto *const atomic_data = ionization_states->atomic_data();
  const auto mass_fractions = comps->mass_fractions();
  const auto species = comps->charge();
  auto ionization_fractions = ionization_states->ionization_fractions();

  // pull out atomic data containers
  const auto ion_data = atomic_data->ion_data();
  const auto species_offsets = atomic_data->offsets();

  static constexpr int ilo = 0;
  const auto &ihi = grid.get_ihi() + 1;
  const auto &nNodes = grid.get_n_nodes();
  assert(ionization_fractions.extent(2) <=
         static_cast<size_t>(std::numeric_limits<int>::max()));
  const auto &ncomps = static_cast<int>(ionization_fractions.extent(2));

  Kokkos::parallel_for(
      "Saha :: Solve Ionization All",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ilo, 0, 0},
                                             {ihi + 1, nNodes + 2, ncomps}),
      KOKKOS_LAMBDA(const int ix, const int node, const int e) {
        auto lambda = nullptr;
        const double tau = fluid_basis.basis_eval(uCF, ix, 0, node);
        const double vel = fluid_basis.basis_eval(uCF, ix, 1, node);
        const double emt = fluid_basis.basis_eval(uCF, ix, 2, node);
        const double temperature =
            temperature_from_conserved(&eos, tau, vel, emt, lambda);

        const double x_e = fluid_basis.basis_eval(mass_fractions, ix, e, node);

        const int z = e + 1;
        const double nk = element_number_density(x_e, z, 1.0 / tau);

        // pull out element info
        const auto species_atomic_data =
            species_data(ion_data, species_offsets, e);
        auto ionization_fractions_e =
            Kokkos::subview(ionization_fractions, ix, node, e, Kokkos::ALL);

        saha_solve(ionization_fractions_e, z, temperature, species_atomic_data,
                   nk);
      });
}

KOKKOS_FUNCTION
auto saha_f(const double T, const IonLevel &ion_data) -> double {
  const double prefix = 2.0 * (ion_data.g_upper / ion_data.g_lower) *
                        constants::k_saha * std::pow(T, 1.5);
  const double suffix = std::exp(-ion_data.chi / (constants::k_Bev * T));

  return prefix * suffix;
}

/**
 * @brief Compute neutral ionization fraction. Eq 8 of Zaghloul et al 2000.
 */
KOKKOS_FUNCTION
auto ion_frac0(const double Zbar, const double temperature,
               const View1D<const IonLevel> ion_datas, const double nh,
               const int min_state, const int max_state) -> double {

  double denominator = 0.0;
  for (int i = min_state; i < max_state; ++i) {
    double inner_num = 1.0;
    for (int j = min_state; j <= i; ++j) {
      inner_num *= (i * saha_f(temperature, ion_datas(j - 1)));
    }
    denominator += inner_num / std::pow(Zbar * nh, i);
  }
  denominator += (min_state - 1.0);
  return Zbar / denominator;
}

KOKKOS_FUNCTION
void saha_solve(View1D<double> ionization_states, const int Z,
                const double temperature,
                const View1D<const IonLevel> ion_datas, const double nk) {

  const int num_states = Z + 1;
  int min_state = 1;
  int max_state = num_states;

  const double Zbar_nk_inv = 1.0 / (Z * nk);

  for (int i = 0; i < num_states - 1; ++i) {
    const double f_saha = std::abs(saha_f(temperature, ion_datas(i)));

    if (f_saha * Zbar_nk_inv > root_finders::ZBARTOLINV) {
      min_state = i + 1;
      ionization_states(i) = 0.0;
    }
    if (f_saha * Zbar_nk_inv < root_finders::ZBARTOL) {
      max_state = i;
      for (int j = i + 1; j < num_states; ++j) {
        ionization_states(j) = 0.0;
      }
      break;
    }
  }

  double Zbar = 0;
  if (max_state == 0) {
    ionization_states(0) = 1.0; // neutral
    Zbar = 1.0e-16; // uncharged (but don't want division by 0)
  } else if (min_state == num_states) {
    ionization_states(0) =
        ion_frac0(Zbar, temperature, ion_datas, nk, min_state, max_state);
    ionization_states(num_states - 1) = 1.0; // full ionization
    Zbar = Z;
  } else if (min_state == max_state) {
    Zbar = min_state - 1.0;
    ionization_states(min_state) = 1.0; // only one state possible
  } else { // iterative solve
    const double guess = 0.5 * Z;

    // we use an Anderson acclerated Newton Raphson iteration
    Zbar =
        root_finders::newton_aa(saha_target, saha_d_target, guess, temperature,
                                ion_datas, nk, min_state, max_state);

    ionization_states(0) =
        ion_frac0(Zbar, temperature, ion_datas, nk, min_state, max_state);
    for (int i = 1; i <= Z; ++i) {
      ionization_states(i) =
          ionization_states(i - 1) *
          (saha_f(temperature, ion_datas(i - 1)) / (Zbar * nk));
    }
  }
}

KOKKOS_FUNCTION
auto saha_target(const double Zbar, const double T,
                 const View1D<const IonLevel> ion_datas, const double nh,
                 const int min_state, const int max_state) -> double {
  double result = Zbar;
  double numerator = 1.0;
  double denominator = 0.0;
  for (int i = min_state; i < max_state; ++i) {
    double inner_denom = 1.0;
    double inner_num = 1.0;
    for (int j = min_state; j <= i; ++j) {
      inner_num *= saha_f(T, ion_datas(j - 1));
      inner_denom *= (i * saha_f(T, ion_datas(j - 1)));
    }
    numerator += inner_num / std::pow(Zbar * nh, i);
    denominator += inner_denom / std::pow(Zbar * nh, i);
  }
  denominator += (min_state - 1.0);

  result *= (numerator / denominator);
  result = 1.0 - result;
  return result;
}

KOKKOS_FUNCTION
auto saha_d_target(const double Zbar, const double T,
                   const View1D<const IonLevel> ion_datas, const double nh,
                   const int min_state, const int max_state) -> double {

  double product = 1.0;
  double sigma0 = 0.0;
  double sigma1 = 0.0;
  double sigma2 = 0.0;
  double sigma3 = 0.0;

  for (int i = min_state; i < max_state; ++i) {
    product *= saha_f(T, ion_datas(i - 1)) / (Zbar * nh);
    sigma0 += product;
    sigma1 += i * product;
    sigma2 += (i - min_state + 1.0) * product;
    sigma3 += i * (i - min_state + 1.0) * product;
  }

  const double denom = 1.0 / (min_state - 1.0 + sigma1);
  return (sigma2 - (1.0 + sigma0) * (1.0 + sigma3 * denom)) * denom;
}
