#pragma once
/**
 * @file rad_utilities.cpp
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
 *          - computeTimestep_Rad
 */

#include <tuple>

#include "radiation/radhydro_package.hpp"
#include "solvers/root_finder_opts.hpp"
#include "solvers/root_finders.hpp"

namespace radiation {

using root_finders::PhysicalScales, root_finders::RadHydroConvergence;

auto flux_factor(double E, double F) -> double;
auto flux_rad(double E, double F, double P, double vstar, int iCR) -> double;
auto flux_rad(double E, double F, double P, double V)
    -> std::tuple<double, double>;
auto radiation_four_force(double D, double V, double T, double kappa_r,
                          double kappa_p, double E, double F, double Pr)
    -> std::tuple<double, double>;
auto source_factor_rad() -> std::tuple<double, double>;
auto compute_closure(double E, double F) -> double;
auto lambda_hll(double f, int sign) -> double;
auto llf_flux(double Fp, double Fm, double Up, double Um, double alpha)
    -> double;
auto numerical_flux_hll_rad(double E_L, double E_R, double F_L, double F_R,
                            double P_L, double P_R, double vstar)
    -> std::tuple<double, double>;

// Custom root finder for radiation-matter coulpling.
// This should not live here forever.
// TODO(astrobarker): port to the new root finders infra
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

} // namespace radiation
