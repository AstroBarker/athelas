#pragma once
/**
 * @file bound_enforcing_limiter.hpp
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
 *          and an Anderson accelerated fixed point iteration is the default.
 *          point iteration being the default choice.
 */

#include <print>

#include "abstractions.hpp"
#include "polynomial_basis.hpp"
#include "utils/utilities.hpp"

namespace bel {

void limit_density(View3D<double> U, const ModalBasis* basis);
void limit_internal_energy(View3D<double> U, const ModalBasis* basis);
void limit_rad_energy(View3D<double> U, const ModalBasis* basis);
void limit_rad_momentum(View3D<double> U, const ModalBasis* basis);
void apply_bound_enforcing_limiter(View3D<double> U, const ModalBasis* basis);
void apply_bound_enforcing_limiter_rad(View3D<double> U,
                                       const ModalBasis* basis);
auto compute_theta_state(View3D<double> U, const ModalBasis* basis,
                         double theta, int iCF, int iX, int iN)
    -> double;
auto target_func(double theta, View3D<double> U, const ModalBasis* basis,
                 int iX, int iN) -> double;
auto target_func_rad_flux(double theta, View3D<double> U,
                          const ModalBasis* basis, int iX,
                          int iN) -> double;
auto target_func_rad_energy(double theta, View3D<double> U,
                            const ModalBasis* basis, int iX,
                            int iN) -> double;

template <typename F>
auto bisection(const View3D<double> U, F target, const ModalBasis* basis,
               const int iX, const int iN) -> double {
  constexpr static double TOL    = 1e-10;
  constexpr static int MAX_ITERS = 100;
  constexpr static double delta  = 1.0e-3; // reduce root by delta

  // bisection bounds on theta
  double a = 0.0;
  double b = 1.0;
  double c = 0.5;

  double fa = 0.0; // f(a) etc
  double fc = 0.0;

  int n = 0;
  while (n <= MAX_ITERS) {
    c = (a + b) / 2.0;

    fa = target(a, U, basis, iX, iN);
    fc = target(c, U, basis, iX, iN);

    if (std::abs(fc) <= TOL || (b - a) / 2.0 < TOL) {
      return c - delta;
    }

    // new interval
    if (utilities::SGN(fc) == utilities::SGN(fa)) {
      a = c;
    } else {
      b = c;
    }

    n++;
  }

  std::println("Max Iters Reach In bisection");
  return c - delta;
}

template <typename F>
auto backtrace(const View3D<double> U, F target, const ModalBasis* basis,
               const int iX, const int iN) -> double {
  constexpr static double EPSILON = 1.0e-10; // maybe make this smarter
  double theta                    = 1.0;
  double nodal                    = -1.0;

  while (theta >= 0.01 && nodal < EPSILON) {
    nodal = target(theta, U, basis, iX, iN);

    theta -= 0.05;
  }

  return theta;
}

} // namespace bel
