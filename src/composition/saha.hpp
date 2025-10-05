#pragma once

#include "atom/atom.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "polynomial_basis.hpp"
#include "state/state.hpp"

namespace athelas::atom {

void solve_saha_ionization(State &state, const GridStructure &grid,
                           const eos::EOS &eos,
                           const basis::ModalBasis &fluid_basis);
KOKKOS_INLINE_FUNCTION
auto saha_f(const double T, const IonLevel &ion_data) -> double {
  const double prefix = 2.0 * (ion_data.g_upper / ion_data.g_lower) *
                        constants::k_saha * std::pow(T, 1.5);
  const double suffix = std::exp(-ion_data.chi / (constants::k_Bev * T));

  return prefix * suffix;
}

/**
 * @brief Compute neutral ionization fraction. Eq 8 of Zaghloul et al 2000.
 */
KOKKOS_INLINE_FUNCTION
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

KOKKOS_INLINE_FUNCTION
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

KOKKOS_INLINE_FUNCTION
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

KOKKOS_INLINE_FUNCTION
void saha_solve(View1D<double> ionization_states, const int Z,
                const double temperature,
                const View1D<const IonLevel> ion_datas, const double nk) {

  using root_finders::RootFinder, root_finders::AANewtonAlgorithm;
  // Set up static root finder for Saha ionization
  // We keep tight tolerances here.
  // TODO(astrobarker): make tolerances runtime
  static RootFinder<double, AANewtonAlgorithm<double>> solver(
      {.abs_tol = 1.0e-16, .rel_tol = 1.0e-14, .max_iterations = 100});
  static constexpr double ZBARTOL = 1.0e-15;
  static constexpr double ZBARTOLINV = 1.0e15;

  const int num_states = Z + 1;
  int min_state = 1;
  int max_state = num_states;

  const double Zbar_nk_inv = 1.0 / (Z * nk);

  for (int i = 0; i < num_states - 1; ++i) {
    const double f_saha = std::abs(saha_f(temperature, ion_datas(i)));

    if (f_saha * Zbar_nk_inv > ZBARTOLINV) {
      min_state = i + 1;
      ionization_states(i) = 0.0;
    }
    if (f_saha * Zbar_nk_inv < ZBARTOL) {
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
    // I wonder if there is a smarter way to produce a guess -- T dependent?
    // Simpler ionization model to guess Zbar(T)?
    const double guess = 0.5 * Z;

    // we use an Anderson acclerated Newton Raphson iteration
    Zbar = solver.solve(saha_target, saha_d_target, guess, temperature,
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

} // namespace athelas::atom
