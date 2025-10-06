/**
 * @file slope_limiter_utilities.hpp
 * --------------
 *
 * @brief Utility functions for slope limiters.
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "limiters/slope_limiter.hpp"
#include "utils/utilities.hpp"

namespace athelas {

using namespace utilities;

auto initialize_slope_limiter(std::string field, const GridStructure *grid,
                              const ProblemIn *pin,
                              const std::vector<int> &vars, int nvars)
    -> SlopeLimiter;

// Standard MINMOD function
template <typename T>
constexpr auto MINMOD(T a, T b, T c) -> T {
  if (SGN(a) == SGN(b) && SGN(b) == SGN(c)) {
    return SGN(a) * std::min({std::abs(a), std::abs(b), std::abs(c)});
  }
  return T(0);
}

// TVB MINMOD function
template <typename T>
constexpr auto MINMOD_B(T a, T b, T c, T dx, T M) -> T {
  if (std::abs(a) > M * dx * dx) {
    return MINMOD(a, b, c);
  }
  return a;
}

auto barth_jespersen(double U_v_L, double U_v_R, double U_c_L, double U_c_T,
                     double U_c_R, double alpha) -> double;

void detect_troubled_cells(const AthelasArray3D<double> U,
                           AthelasArray1D<double> D, const GridStructure *grid,
                           const basis::ModalBasis *basis,
                           const std::vector<int> &vars);

auto cell_average(AthelasArray3D<double> U, const GridStructure *grid,
                  const basis::ModalBasis *basis, int q, int ix,
                  int extrapolate) -> double;

void modify_polynomial(AthelasArray3D<double> U,
                       AthelasArray2D<double> modified_polynomial,
                       double gamma_i, double gamma_l, double gamma_r, int ix,
                       int q);

auto smoothness_indicator(AthelasArray3D<double> U,
                          AthelasArray2D<double> modified_polynomial,
                          const GridStructure *grid,
                          const basis::ModalBasis *basis, int ix, int i,
                          int iCQ) -> double;

auto non_linear_weight(double gamma, double beta, double tau, double eps)
    -> double;

auto weno_tau(double beta_l, double beta_i, double beta_r, double weno_r)
    -> double;
} // namespace athelas
