/**
 * @file bound_enforcing_limiter.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Implementation of bound enforcing limiters for enforcing physicality.
 *
 * @details This file implements a suite of bound enforcing limiters based on
 *          K. Schaal et al 2015 (ADS: 10.1093/mnras/stv1859). These limiters
 *          ensure physicality of the solution by preventing negative values of
 *          key physical quantities:
 *
 *          - limit_density: Prevents negative density by scaling slope
 *            coefficients
 *          - limit_internal_energy: Maintains positive internal energy using
 *            root-finding algorithms
 *          - limit_rad_momentum: Ensures physical radiation momentum values
 *
 *          Multiple root finders for the internal energy solve are implemented
 *          and an Anderson accelerated newton iteration is the default.
 */

#include <algorithm> // std::min, std::max
#include <cmath>
#include <cstdlib> /* abs */

#include "basis/polynomial_basis.hpp"
#include "limiters/bound_enforcing_limiter.hpp"
#include "solvers/root_finders.hpp"
#include "utils/utilities.hpp"

namespace bel {

/**
 * @brief Limits density to maintain physicality following K. Schaal et al 2015
 *
 * @details This function implements the density limiter based on K. Schaal et
 * al 2015 (ADS: 10.1093/mnras/stv1859). It finds a scaling factor theta that
 * ensures density remains positive by computing: theta = min((rho_avg -
 * eps)/(rho_nodal - rho_avg), 1)
 *
 * @param U The solution array containing conserved variables
 * @param basis The modal basis used for the solution representation
 */
void limit_density(View3D<double> U, const ModalBasis* basis) {
  constexpr static double EPSILON = 1.0e-10; // maybe make this smarter

  const int order = basis->get_order();

  if (order == 1) {
    return;
  }

  Kokkos::parallel_for(
      "BEL::Limit Density", Kokkos::RangePolicy<>(1, U.extent(0) - 1),
      KOKKOS_LAMBDA(const int iX) {
        double theta1 = 100000.0; // big
        double nodal = 0.0;
        double frac = 0.0;
        const double avg = U(iX, 0, 0);

        for (int iN = 0; iN <= order; iN++) {
          nodal = basis->basis_eval(U, iX, 0, iN);
          if (std::isnan(nodal)) {
            theta1 = 0.0;
            break;
          }
          frac = std::abs((avg - EPSILON) / (avg - nodal));
          theta1 = std::min(theta1, std::min(1.0, frac));
        }

        for (int k = 1; k < order; k++) {
          U(iX, k, 0) *= theta1;
        }
      });
}

/**
 * @brief Limits the solution to maintain positivity of internal energy
 *
 * @details This function implements the bound enforcing limiter for internal
 *          energy based on K. Schaal et al 2015
 *          (ADS: 10.1093/mnras/stv1859). It finds a scaling factor theta such
 *          that (1 - theta) * U_bar + theta * U_q is positive for U being the
 *          specific internal energy.
 *
 *          The function uses three possible root finding algorithms:
 *          - bisection: A robust but slower method
 *          - Anderson accelerated newton iteration: The default method
 *          - Back tracing: A simple algorithm that steps back from theta = 1
 *
 *          All methods yield the same results and are stable on difficult
 *          problems. The simulation time is insensitive to solver choice.
 *
 *          Note: In the bisection and fixed point solvers, a small delta =
 *          1.0e-3 is subtracted from the root to ensure positivity. The
 *          backtrace algorithm overshoots the root, so this adjustment is not
 *          necessary.
 *
 * @param U The solution array containing conserved variables
 * @param basis The modal basis used for the solution representation
 */
void limit_internal_energy(View3D<double> U, const ModalBasis* basis) {
  constexpr static double EPSILON = 1.0e-10; // maybe make this smarter

  const int order = basis->get_order();

  if (order == 1) {
    return;
  }

  Kokkos::parallel_for(
      "BEL::Limit Internal Energy", Kokkos::RangePolicy<>(1, U.extent(0) - 2),
      KOKKOS_LAMBDA(const int iX) {
        double theta2 = 10000000.0;
        double nodal = 0.0;
        double temp = 0.0;

        for (int iN = 0; iN <= order + 1; iN++) {
          nodal = utilities::compute_internal_energy(U, basis, iX, iN);

          if (nodal > EPSILON) {
            temp = 1.0;
          } else {
            const double theta_guess = 0.9;
            //temp = bisection(U, target_func, basis, iX, iN);
            temp = root_finders::newton_aa(target_func, target_func_deriv, theta_guess, U, basis, iX, iN);
          }
          theta2 = std::min(theta2, temp);
        }

        for (int k = 1; k < order; k++) {
          U(iX, k, 0) *= theta2;
          U(iX, k, 1) *= theta2;
          U(iX, k, 2) *= theta2;
        }
      });
}

void apply_bound_enforcing_limiter(View3D<double> U, const ModalBasis* basis)

{
  limit_density(U, basis);
  limit_internal_energy(U, basis);
}

// TODO(astrobarker): much more here.
void apply_bound_enforcing_limiter_rad(View3D<double> U,
                                       const ModalBasis* basis) {
  if (basis->get_order() == 1) {
    return;
  }
  limit_rad_energy(U, basis);
  //limit_rad_momentum(U, basis);
}

void limit_rad_energy(View3D<double> U, const ModalBasis* basis) {
  constexpr static double EPSILON = 1.0e-4; // maybe make this smarter

  const int order = basis->get_order();

  Kokkos::parallel_for(
      "BEL::Limit Rad Energy", Kokkos::RangePolicy<>(1, U.extent(0) - 1),
      KOKKOS_LAMBDA(const int iX) {
        double theta2 = 10000000.0;
        double nodal = 0.0;
        double temp = 0.0;

        for (int iN = 0; iN <= order + 1; iN++) {
          nodal = basis->basis_eval(U, iX, 3, iN);

          if (nodal > EPSILON + 0 * std::abs(U(iX, 0, 4)) / constants::c_cgs) {
            temp = 1.0;
          } else {
            const double theta_guess = 0.9;
            //temp = bisection(U, target_func_rad_energy, basis, iX, iN);
            temp = root_finders::newton_aa(target_func_rad_energy, target_func_rad_energy_deriv, theta_guess, U, basis, iX, iN);
          }
          theta2 = std::abs(std::min(theta2, temp));
        }

        for (int k = 1; k < order; k++) {
          U(iX, k, 3) *= theta2;
          U(iX, k, 4) *= theta2;
        }
      });
}

void limit_rad_momentum(View3D<double> U, const ModalBasis* basis) {
  const int order = basis->get_order();

  Kokkos::parallel_for(
      "BEL::Limit Rad Momentum", Kokkos::RangePolicy<>(1, U.extent(2) - 1),
      KOKKOS_LAMBDA(const int iX) {
        double theta2 = 10000000.0;
        double nodal = 0.0;
        double temp = 0.0;

        constexpr static double c = constants::c_cgs;

        for (int iN = 0; iN <= order + 1; iN++) {
          nodal = basis->basis_eval(U, iX, 4, iN);

          if (std::abs(nodal) <= c * U(iX, 0, 3)) {
            temp = 1.0;
          } else {
             const double theta_guess = 0.9;
             temp = root_finders::newton_aa(target_func_rad_flux, target_func_rad_flux_deriv, theta_guess, U, basis, iX, iN) - 1.0e-3;
            // temp = bisection(U, target_func_rad_flux, basis, iX, iN);
          }
          theta2 = std::abs(std::min(theta2, temp));

        }

        for (int k = 1; k < order; k++) {
          U(iX, k, 4) *= theta2;
        }
      });
}

/* --- Utility Functions --- */

// ( 1 - theta ) U_bar + theta U_q
auto compute_theta_state(const View3D<double> U, const ModalBasis* basis,
                         const double theta, const int iCF, const int iX,
                         const int iN) -> double {
  return theta * (basis->basis_eval(U, iX, iCF, iN) - U(iX, 0, iCF)) + U(iX, 0, iCF);
}

auto target_func(const double theta, const View3D<double> U,
                 const ModalBasis* basis, const int iX, const int iN)
    -> double {
  const double w = 1.0e-13;
  const double s1 = compute_theta_state(U, basis, theta, 1, iX, iN);
  const double s2 = compute_theta_state(U, basis, theta, 2, iX, iN);

  double const e = s2 - (0.5 * s1 * s1);

  return e - w;
}
auto target_func_deriv(const double theta, const View3D<double> U, const ModalBasis* basis,
                 const int iX, const int iN) -> double {
  const double dE = basis->basis_eval(U, iX, 2, iN) - U(0, iX, 2);
  const double v_q = basis->basis_eval(U, iX, 1, iN);
  const double dv = v_q - U(0, iX, 1);
  return dE - (v_q + theta * dv) * dv;
}

// TODO(astrobarker) some redundancy below
auto target_func_rad_flux(const double theta, const View3D<double> U,
                          const ModalBasis* basis, const int iX, const int iN)
    -> double {
  const double w = 1.0e-13; 
  const double s1 = compute_theta_state(U, basis, theta, 4, iX, iN);
  const double s2 = compute_theta_state(U, basis, theta, 3, iX, iN);

  const double e = std::abs(s1) / (constants::c_cgs * s2);

  return e - w;
}

auto target_func_rad_flux_deriv(const double theta, const View3D<double> U,
                          const ModalBasis* basis, const int iX, const int iN)
    -> double {
  const double dE = basis->basis_eval(U, iX, 3, iN) - U(3, iX, 0);
  const double dF = basis->basis_eval(U, iX, 4, iN) - U(4, iX, 0);
  const double E_theta = compute_theta_state(U, basis, theta, 3, iX, iN);
  const double F_theta = compute_theta_state(U, basis, theta, 4, iX, iN);
  const double dfdE = - F_theta / (E_theta * E_theta * constants::c_cgs);
  const double dfdF = F_theta / (std::abs(F_theta) * E_theta * constants::c_cgs);
  return dfdE * dE + dfdF * dF;
}

auto target_func_rad_energy_deriv(const double theta, const View3D<double> U, const ModalBasis* basis,
                 const int iX, const int iN) -> double {
  return basis->basis_eval(U, iX, 3, iN) - U(0, iX, 3);
}

auto target_func_rad_energy(const double theta, const View3D<double> U,
                            const ModalBasis* basis, const int iX, const int iN)
    -> double {
  const double w = 1.0e-13;
  const double s1 = compute_theta_state(U, basis, theta, 3, iX, iN);

  const double e = s1;

  return e - w;
}

} // namespace bel
