#ifndef SLOPE_LIMITER_UTILITIES_HPP_
#define SLOPE_LIMITER_UTILITIES_HPP_
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

auto InitializeSlopeLimiter( const GridStructure* grid, const ProblemIn* pin,
                             int nvars ) -> SlopeLimiter;

// Standard minmod function
template <typename T>
constexpr auto minmod( T a, T b, T c ) -> Real {
  if ( sgn( a ) == sgn( b ) && sgn( b ) == sgn( c ) ) {
    return sgn( a ) *
           std::min( std::min( std::abs( a ), std::abs( b ) ), std::abs( c ) );
  }
  return 0.0;
}

// TVB minmod function
template <typename T>
constexpr auto minmodB( T a, T b, T c, T dx, T M ) -> Real {
  if ( std::abs( a ) < M * dx * dx ) {
    return a;
  }
  return minmod( a, b, c );
}

auto BarthJespersen( Real U_v_L, Real U_v_R, Real U_c_L, Real U_c_T, Real U_c_R,
                     Real alpha ) -> Real;

void DetectTroubledCells( View3D<Real> U, View2D<Real> D,
                          const GridStructure* Grid, const ModalBasis* Basis );

auto CellAverage( View3D<Real> U, const GridStructure* Grid,
                  const ModalBasis* Basis, int iCF, int iX, int extrapolate )
    -> Real;

void ModifyPolynomial( View3D<Real> U, View2D<Real> modified_polynomial,
                       Real gamma_i, Real gamma_l, Real gamma_r, int iX,
                       int iCQ );

auto SmoothnessIndicator( View3D<Real> U, View2D<Real> modified_polynomial,
                          const GridStructure* Grid, int iX, int i, int iCQ )
    -> Real;

auto NonLinearWeight( Real gamma, Real beta, Real tau, Real eps ) -> Real;

auto Tau( Real beta_l, Real beta_i, Real beta_r, Real weno_r ) -> Real;
} // namespace limiter_utilities
#endif // SLOPE_LIMITER_UTILITIES_HPP_
