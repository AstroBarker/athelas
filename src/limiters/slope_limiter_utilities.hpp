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

auto initialize_slope_limiter( const GridStructure* grid, const ProblemIn* pin,
                               int nvars ) -> SlopeLimiter;

// Standard MINMOD function
template <typename T>
constexpr auto MINMOD( T a, T b, T c ) -> T {
  if ( SGN( a ) == SGN( b ) && SGN( b ) == SGN( c ) ) {
    return SGN( a ) *
           std::min({std::abs(a), std::abs(b), std::abs(c)});
  }
  return T(0);
}

// TVB MINMOD function
template <typename T>
constexpr auto MINMOD_B( T a, T b, T c, T dx, T M ) -> T {
  if ( std::abs( a ) > M * dx * dx ) {
    return MINMOD(a, b, c);
  }
  return a;
}

auto barth_jespersen( Real U_v_L, Real U_v_R, Real U_c_L, Real U_c_T,
                      Real U_c_R, Real alpha ) -> Real;

void detect_troubled_cells( View3D<Real> U, View2D<Real> D,
                            const GridStructure* grid,
                            const ModalBasis* basis );

auto cell_average( View3D<Real> U, const GridStructure* grid,
                   const ModalBasis* basis, int iCF, int iX, int extrapolate )
    -> Real;

void modify_polynomial( View3D<Real> U, View2D<Real> modified_polynomial,
                        Real gamma_i, Real gamma_l, Real gamma_r, int iX,
                        int iCQ );

auto smoothness_indicator( View3D<Real> U, View2D<Real> modified_polynomial,
                           const GridStructure* grid, const ModalBasis* basis,
                           int iX, int i, int iCQ )
    -> Real;

auto non_linear_weight( Real gamma, Real beta, Real tau, Real eps ) -> Real;

auto weno_tau( Real beta_l, Real beta_i, Real beta_r, Real weno_r ) -> Real;
} // namespace limiter_utilities
