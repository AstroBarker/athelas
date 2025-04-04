#ifndef SLOPE_LIMITER_UTILITIES_HPP_
#define SLOPE_LIMITER_UTILITIES_HPP_

#include "abstractions.hpp"
#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "slope_limiter.hpp"
#include "utils/utilities.hpp"

using namespace utilities;

SlopeLimiter InitializeSlopeLimiter(const GridStructure *grid, const ProblemIn *pin, const int nvars);

// Standard minmod function
template <typename T>
constexpr Real minmod( T a, T b, T c ) {
  if ( sgn( a ) == sgn( b ) && sgn( b ) == sgn( c ) ) {
    return sgn( a ) *
           std::min( std::min( std::abs( a ), std::abs( b ) ), std::abs( c ) );
  } else {
    return 0.0;
  }
}

// TVB minmod function
template <typename T>
constexpr Real minmodB( T a, T b, T c, T dx, T M ) {
  if ( std::abs( a ) < M * dx * dx ) {
    return a;
  } else {
    return minmod( a, b, c );
  }
}

Real BarthJespersen( Real U_v_L, Real U_v_R, Real U_c_L, Real U_c_T, Real U_c_R,
                     Real alpha );

void DetectTroubledCells( View3D<Real> U, View2D<Real> D,
                          const GridStructure *Grid, const ModalBasis *Basis );

Real CellAverage( View3D<Real> U, const GridStructure *Grid,
                  const ModalBasis *Basis, const int iCF, const int iX,
                  const int extrapolate );

void ModifyPolynomial( const View3D<Real> U, View2D<Real> modified_polynomial,
                       const Real gamma_i, const Real gamma_l,
                       const Real gamma_r, const int iX, const int iCQ );

Real SmoothnessIndicator( const View3D<Real> U,
                          const View2D<Real> modified_polynomial,
                          const GridStructure *Grid, const int iX, const int i,
                          const int iCQ );

Real NonLinearWeight( const Real gamma, const Real beta, const Real tau,
                      const Real eps );

Real Tau( const Real beta_l, const Real beta_i, const Real beta_r,
          const Real weno_r );
#endif // SLOPE_LIMITER_UTILITIES_HPP_
