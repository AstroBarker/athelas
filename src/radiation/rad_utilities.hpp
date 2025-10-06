/**
 * @file rad_utilities.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Functions for radiation evolution.
 *
 * @details Key functions for radiation udates:
 *          - flux_factor
 *          - flux_rad
 *          - radiation_four_force
 *          - source_rad
 *          - Compute_Closure
 *          - lambda_hll
 *          - numerical_flux_hll_rad
 */

#pragma once

#include <tuple>

#include "Kokkos_Macros.hpp"
#include "radiation/radhydro_package.hpp"
#include "solvers/root_finder_opts.hpp"
#include "solvers/root_finders.hpp"
#include "utils/riemann.hpp"

namespace athelas::radiation {

using root_finders::PhysicalScales, root_finders::RadHydroConvergence;

/**
 * @brief radiation flux factor
 **/
KOKKOS_FORCEINLINE_FUNCTION
auto flux_factor(const double E, const double F) -> double {
  assert(E > 0.0 &&
         "Radiation :: flux_factor :: non positive definite energy density.");
  return std::abs(F) / (constants::c_cgs * E);
}

/**
 * @brief return std::tuple containing advective radiation flux
 */
KOKKOS_INLINE_FUNCTION
auto flux_rad(const double E, const double F, const double P, const double V)
    -> std::tuple<double, double> {
  return {F - E * V, constants::c_cgs * constants::c_cgs * P - F * V};
}

/**
 * @brief Radiation 4 force for rad-matter interactions
 * Assumes kappa_e ~ kappa_p, kappa_F ~ kappa_r
 * D : Density
 * V : Velocity
 * T : Temperature
 * kappa_r : rosseland kappa
 * kappa_p : planck kappa
 * E : radiation energy density
 * F : radiation momentum density
 * Pr : radiation momentum closure
 **/
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto
radiation_four_force(const double D, const double V, const double T,
                     const double kappa_r, const double kappa_p, const double E,
                     const double F, const double Pr)
    -> std::tuple<double, double> {
  assert(D >= 0.0 &&
         "Radiation :: RadiationFourFource :: Non positive definite density.");
  assert(T > 0.0 &&
         "Radiation :: RadiationFourFource :: Non positive temperature.");
  assert(E > 0.0 && "Radiation :: RadiationFourFource :: Non positive "
                    "definite radiation energy density.");

  constexpr static double a = constants::a;
  constexpr static double c = constants::c_cgs;

  const double b = V / c;
  const double term1 = E - (a * T * T * T * T);
  const double Fc = F / c;

  // O(b^2) ala Fuksman
  /*
  const double kappa = kappa_r;
  const double G0 = D * kappa * ( term1 - b * Fc - b * b * E - b * b * Pr );
  const double G  = D * kappa * ( b * ( term1 - 2.0 * b * Fc ) + ( Fc - b * E -
  b * Pr ) );
  */

  // Krumholz et al. 2007 O(b^2)
  const double G0 =
      D * (kappa_p * term1 + (kappa_r - 2.0 * kappa_p) * b * Fc +
           0.5 * (2.0 * (kappa_p - kappa_r) * E + kappa_p * term1) * b * b +
           (kappa_p - kappa_r) * b * b * Pr);

  const double G =
      D * (kappa_r * Fc + kappa_p * term1 * b - kappa_r * b * (E + Pr) +
           0.5 * kappa_r * Fc * b * b + 2.0 * (kappa_r - kappa_p) * b * b * Fc);

  // ala Skinner & Ostriker, simpler.
  /*
  const double kappa = kappa_r;
  const double G0 = D * kappa * ( term1 - b * Fc );
  const double G  = D * kappa * ( Fc - b * E + b * Pr );
  */
  return {G0, G};
}

/**
 * @brief factor of c scaling terms for radiation-matter sources
 **/
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto source_factor_rad()
    -> std::tuple<double, double> {
  constexpr static double c = constants::c_cgs;
  return {c, c * c};
}

/**
 * @brief M1 closure of Levermore 1984
 * TODO(astrobarker): It would be nice to make this easier to modify
 * Perhaps CRTP model
 */
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto compute_closure(const double E,
                                                          const double F)
    -> double {
  assert(E > 0.0 && "Radiation :: compute_closure :: Non positive definite "
                    "radiation energy density.");
  constexpr static double one_third = 1.0 / 3.0;
  const double f = std::clamp(flux_factor(E, F), 0.0, 1.0);
  const double f2 = f * f;
  const double chi =
      (3.0 + 4.0 * f2) / (5.0 + 2.0 * std::sqrt(4.0 - (3.0 * f2)));
  const double T = std::clamp(
      ((1.0 - chi) / 2.0) + ((3.0 * chi - 1.0) * 1.0 / 2.0), one_third, 1.0);
  return E * T;
}

/**
 * @brief LLF numerical flux
 */
auto KOKKOS_FORCEINLINE_FUNCTION llf_flux(const double Fp, const double Fm,
                                          const double Up, const double Um,
                                          const double alpha) -> double {
  return 0.5 * std::fma(alpha, (Um - Up), (Fp + Fm));
}

/**
 * @brief eigenvalues of Jacobian for radiation solve
 * see 2013ApJS..206...21S (Skinner & Ostriker 2013) Eq 41a,b
 * and references therein
 **/
auto lambda_hll(const double f, const int sign) -> double {
  constexpr static double c = constants::c_cgs;
  constexpr static double twothird = 2.0 / 3.0;

  const double f2 = f * f;
  const double sqrtterm = std::sqrt(4.0 - (3.0 * f2));
  auto res = c *
             (f + sign * std::sqrt((twothird * (4.0 - 3.0 * f2 - sqrtterm)) +
                                   (2.0 * (2.0 - f2 - sqrtterm)))) /
             sqrtterm;
  return res;
}

/**
 * @brief HLL Riemann solver for radiation
 * see 2013ApJS..206...21S (Skinner & Ostriker 2013) Eq 39
 * and references & discussion therein
 **/
auto numerical_flux_hll_rad(const double E_L, const double E_R,
                            const double F_L, const double F_R,
                            const double P_L, const double P_R,
                            const double vstar) -> std::tuple<double, double> {
  using namespace riemann;

  // flux factors
  const double f_L = flux_factor(E_L, F_L);
  const double f_R = flux_factor(E_R, F_R);

  // TODO(astrobarker) - vstar?
  constexpr static double c2 = constants::c_cgs * constants::c_cgs;
  const double lambda1_L = lambda_hll(f_L, -1.0);
  const double lambda1_R = lambda_hll(f_R, -1.0);
  const double lambda3_L = lambda_hll(f_L, 1.0);
  const double lambda3_R = lambda_hll(f_R, 1.0);
  const double lambda_min_L = lambda1_L;
  const double lambda_min_R = lambda1_R;
  const double lambda_max_L = lambda3_L;
  const double lambda_max_R = lambda3_R;

  const double s_r = std::max(lambda_max_L, lambda_max_R) - vstar;
  const double s_l = std::min(lambda_min_L, lambda_min_R) - vstar;

  const double s_r_p = std::max(s_r, 0.0);
  const double s_l_m = std::min(s_l, 0.0);

  const double flux_e = hll(E_L, E_R, F_L, F_R, s_l_m, s_r_p);
  const double flux_f = hll(F_L, F_R, c2 * P_L, c2 * P_R, s_l_m, s_r_p);
  return {flux_e, flux_f};
}

/**
 * @brief Custom root finder for radiation-matter coulpling.
 * This should not live here forever.
 * TODO(astrobarker): port to the new root finders infra
 */
template <typename T, typename... Args>
KOKKOS_INLINE_FUNCTION void fixed_point_radhydro(T R, double dt_a_ii,
                                                 T scratch_n, T scratch_nm1,
                                                 T scratch, Args... args) {
  static_assert(T::rank == 2, "fixed_point_radhydro expects rank-2 views.");
  static constexpr int nvars = 5;

  const int num_modes = scratch_n.extent(0);

  auto target = [&](T u, const int k) {
    const auto [s_1_k, s_2_k, s_3_k, s_4_k] =
        compute_increment_radhydro_source(u, k, args...);
    return std::make_tuple(R(k, 1) + dt_a_ii * s_1_k, R(k, 2) + dt_a_ii * s_2_k,
                           R(k, 3) + dt_a_ii * s_3_k,
                           R(k, 4) + dt_a_ii * s_4_k);
  };

  for (int k = 0; k < num_modes; ++k) {
    for (int iC = 0; iC < nvars; ++iC) {
      scratch_nm1(k, iC) = scratch_n(k, iC); // set to initial guess
    }
  }

  // Set up physical scales based on your problem
  PhysicalScales scales{};
  scales.velocity_scale = 1e7; // Typical velocity (cm/s)
  scales.energy_scale = 1e12; // Typical energy density
  scales.rad_energy_scale = 1e12; // Typical radiation energy density
  scales.rad_flux_scale = 1e20; // Typical radiation flux

  static RadHydroConvergence<T> convergence_checker(
      scales, root_finders::ABSTOL, root_finders::RELTOL, num_modes);

  unsigned int n = 0;
  bool converged = false;
  while (n <= root_finders::MAX_ITERS && !converged) {
    for (int k = 0; k < num_modes; ++k) {
      const auto [xkp1_1_k, xkp1_2_k, xkp1_3_k, xkp1_4_k] =
          target(scratch_n, k);
      scratch(k, 1) = xkp1_1_k; // fluid vel
      scratch(k, 2) = xkp1_2_k; // fluid energy
      scratch(k, 3) = xkp1_3_k; // rad energy
      scratch(k, 4) = xkp1_4_k; // rad flux

      // --- update ---
      for (int iC = 1; iC < nvars; ++iC) {
        scratch_nm1(k, iC) = scratch_n(k, iC);
        scratch_n(k, iC) = scratch(k, iC);
      }
    }

    converged = convergence_checker.check_convergence(scratch_n, scratch_nm1);
    ++n;
  } // while not converged
}

template <typename T, typename... Args>
KOKKOS_INLINE_FUNCTION void fixed_point_radhydro_aa(T R, double dt_a_ii,
                                                    T scratch_n, T scratch_nm1,
                                                    T scratch, Args... args) {
  static_assert(T::rank == 2, "fixed_point_radhydro expects rank-2 views.");
  constexpr static int nvars = 5;

  const int num_modes = scratch_n.extent(0);

  auto target = [&](T u, const int k) {
    const auto [s_1_k, s_2_k, s_3_k, s_4_k] =
        compute_increment_radhydro_source(u, k, args...);
    return std::make_tuple(R(k, 1) + dt_a_ii * s_1_k, R(k, 2) + dt_a_ii * s_2_k,
                           R(k, 3) + dt_a_ii * s_3_k,
                           R(k, 4) + dt_a_ii * s_4_k);
  };

  // --- first fixed point iteration ---
  for (int k = 0; k < num_modes; ++k) {
    const auto [xnp1_1_k, xnp1_2_k, xnp1_3_k, xnp1_4_k] = target(scratch_n, k);
    scratch(k, 1) = xnp1_1_k;
    scratch(k, 2) = xnp1_2_k;
    scratch(k, 3) = xnp1_3_k;
    scratch(k, 4) = xnp1_4_k;
  }
  for (int k = 0; k < num_modes; ++k) {
    for (int iC = 1; iC < nvars; ++iC) {
      scratch_nm1(k, iC) = scratch_n(k, iC);
      scratch_n(k, iC) = scratch(k, iC);
    }
  }

  // Set up physical scales based on your problem
  PhysicalScales scales{};
  scales.velocity_scale = 1e7; // Typical velocity (cm/s)
  scales.energy_scale = 1e12; // Typical energy density
  scales.rad_energy_scale = 1e12; // Typical radiation energy density
  scales.rad_flux_scale = 1e20; // Typical radiation flux

  static RadHydroConvergence<T> convergence_checker(
      scales, root_finders::ABSTOL, root_finders::RELTOL, num_modes);

  bool converged =
      convergence_checker.check_convergence(scratch_n, scratch_nm1);

  if (converged) {
    return;
  }

  unsigned int n = 1;
  while (n <= root_finders::MAX_ITERS && !converged) {
    for (int k = 0; k < num_modes; ++k) {
      const auto [s_1_n, s_2_n, s_3_n, s_4_n] = target(scratch_n, k);
      const auto [s_1_nm1, s_2_nm1, s_3_nm1, s_4_nm1] = target(scratch_nm1, k);

      // residuals
      const auto r_1_n = residual(s_1_n, scratch_n(k, 1));
      const auto r_2_n = residual(s_2_n, scratch_n(k, 2));
      const auto r_3_n = residual(s_3_n, scratch_n(k, 3));
      const auto r_4_n = residual(s_4_n, scratch_n(k, 4));
      const auto r_1_nm1 = residual(s_1_nm1, scratch_nm1(k, 1));
      const auto r_2_nm1 = residual(s_2_nm1, scratch_nm1(k, 2));
      const auto r_3_nm1 = residual(s_3_nm1, scratch_nm1(k, 3));
      const auto r_4_nm1 = residual(s_4_nm1, scratch_nm1(k, 4));

      // Anderson acceleration alpha
      const auto a_1 = alpha_aa(r_1_n, r_1_nm1);
      const auto a_2 = alpha_aa(r_2_n, r_2_nm1);
      const auto a_3 = alpha_aa(r_3_n, r_3_nm1);
      const auto a_4 = alpha_aa(r_4_n, r_4_nm1);

      // Anderson acceleration update
      const auto xnp1_1_k = a_1 * s_1_nm1 + (1.0 - a_1) * s_1_n;
      const auto xnp1_2_k = a_2 * s_2_nm1 + (1.0 - a_2) * s_2_n;
      const auto xnp1_3_k = a_3 * s_3_nm1 + (1.0 - a_3) * s_3_n;
      const auto xnp1_4_k = a_4 * s_4_nm1 + (1.0 - a_4) * s_4_n;

      scratch(k, 1) = xnp1_1_k; // fluid vel
      scratch(k, 2) = xnp1_2_k; // fluid energy
      scratch(k, 3) = xnp1_3_k; // rad energy
      scratch(k, 4) = xnp1_4_k; // rad flux

      // --- update ---
      for (int iC = 1; iC < nvars; ++iC) {
        scratch_nm1(k, iC) = scratch_n(k, iC);
        scratch_n(k, iC) = scratch(k, iC);
      }
    }

    converged = convergence_checker.check_convergence(scratch_n, scratch_nm1);

    ++n;
  } // while not converged
}

} // namespace athelas::radiation
