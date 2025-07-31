#pragma once
/**
 * @file slope_limiter_utilities.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Utility functions for slope limiters.
 */

#include "abstractions.hpp"
#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "slope_limiter.hpp"
#include "utils/utilities.hpp"

using namespace utilities;

namespace limiter_utilities {

auto initialize_slope_limiter(bool enabled, std::string type, const GridStructure* grid, const ProblemIn* pin,
                              const std::vector<int>& vars, int nvars)
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

void detect_troubled_cells(View3D<double> U, View2D<double> D,
                           const GridStructure* grid, const ModalBasis* basis,
                           const std::vector<int>& vars);

auto cell_average(View3D<double> U, const GridStructure* grid,
                  const ModalBasis* basis, int iCF, int iX, int extrapolate)
    -> double;

void modify_polynomial(View3D<double> U, View2D<double> modified_polynomial,
                       double gamma_i, double gamma_l, double gamma_r, int iX,
                       int iCQ);

auto smoothness_indicator(View3D<double> U, View2D<double> modified_polynomial,
                          const GridStructure* grid, const ModalBasis* basis,
                          int iX, int i, int iCQ) -> double;

auto non_linear_weight(double gamma, double beta, double tau, double eps)
    -> double;

auto weno_tau(double beta_l, double beta_i, double beta_r, double weno_r)
    -> double;
} // namespace limiter_utilities
