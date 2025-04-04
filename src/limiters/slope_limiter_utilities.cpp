/**
 * File     :  slope_limiter_utilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Hold utility functions for the slope limiter class
 *  to keep the class minimal.
 *
 * Contains:
 * ---------
 * BarthJespersen
 **/

#include <algorithm> // std::min, std::max
#include <cstdlib> /* abs */
#include <iostream>
#include <limits>

#include "slope_limiter_utilities.hpp"
#include "utilities.hpp"

/**
 *  UNUSED
 *  Barth-Jespersen limiter
 *  Parameters:
 *  -----------
 *  U_v_*: left/right vertex values on target cell
 *  U_c_*: cell averages of target cell + neighbors
 *    [ left, target, right ]
 *  alpha: scaling coefficient for BJ limiter.
 *    alpha=1 is classical limiter, alpha=0 enforces constant solutions
 **/
Real BarthJespersen( Real U_v_L, Real U_v_R, Real U_c_L, Real U_c_T, Real U_c_R,
                     Real alpha ) {
  // Get U_min, U_max
  Real U_min_L = 10000000.0 * U_c_T;
  Real U_min_R = 10000000.0 * U_c_T;
  Real U_max_L = std::numeric_limits<Real>::epsilon( ) * U_c_T * 0.00001;
  Real U_max_R = std::numeric_limits<Real>::epsilon( ) * U_c_T * 0.00001;

  U_min_L = std::min( U_min_L, std::min( U_c_T, U_c_L ) );
  U_max_L = std::max( U_max_L, std::max( U_c_T, U_c_L ) );
  U_min_R = std::min( U_min_R, std::min( U_c_T, U_c_R ) );
  U_max_R = std::max( U_max_R, std::max( U_c_T, U_c_R ) );

  // loop of cell certices
  Real phi_L = 0.0;
  Real phi_R = 0.0;

  // left vertex
  if ( U_v_L - U_c_T + 1.0 > 1.0 ) {
    phi_L = std::min( 1.0, alpha * ( U_max_L - U_c_T ) / ( U_v_L - U_c_T ) );
  } else if ( U_v_L - U_c_T + 1.0 < 1.0 ) {
    phi_L = std::min( 1.0, alpha * ( U_min_L - U_c_T ) / ( U_v_L - U_c_T ) );
  } else {
    phi_L = 1.0;
  }

  // right vertex
  if ( U_v_R - U_c_T + 1.0 > 1.0 ) {
    phi_R = std::min( 1.0, alpha * ( U_max_R - U_c_T ) / ( U_v_R - U_c_T ) );
  } else if ( U_v_R - U_c_T + 1.0 < 1.0 ) {
    phi_R = std::min( 1.0, alpha * ( U_min_R - U_c_T ) / ( U_v_R - U_c_T ) );
  } else {
    phi_R = 1.0;
  }

  // return min of two values
  return std::min( phi_L, phi_R );
}

/**
 * Apply the Troubled Cell Indicator of Fu & Shu (2017)
 * to flag cells for limiting
 **/
void DetectTroubledCells( View3D<Real> U, View2D<Real> D,
                          const GridStructure *Grid, const ModalBasis *Basis ) {
  const int ilo = Grid->Get_ilo( );
  const int ihi = Grid->Get_ihi( );

  Real denominator = 0.0;

  // Cell averages by extrapolating L and R neighbors into current cell

  // TODO: Kokkos
  for ( int iCF = 0; iCF < 3; iCF++ )
    for ( int iX = ilo; iX <= ihi; iX++ ) {

      if ( iCF == 1 ) continue; /* skip velocity */

      Real result   = 0.0;
      Real cell_avg = U( iCF, iX, 0 );

      // Extrapolate neighboring poly representations into current cell
      // and compute the new cell averages
      Real cell_avg_L_T =
          CellAverage( U, Grid, Basis, iCF, iX + 1, -1 ); // from right
      Real cell_avg_R_T =
          CellAverage( U, Grid, Basis, iCF, iX - 1, +1 ); // from left
      Real cell_avg_L = U( iCF, iX - 1, 0 ); // native left
      Real cell_avg_R = U( iCF, iX + 1, 0 ); // native right

      result += ( std::abs( cell_avg - cell_avg_L_T ) +
                  std::abs( cell_avg - cell_avg_R_T ) );

      denominator =
          std::max( std::max( std::abs( cell_avg_L ), std::abs( cell_avg_R ) ),
                    cell_avg );

      D( iCF, iX ) = result / denominator;
    }
}

/**
 * Return the cell average of a field iCF on cell iX.
 * The parameter `int extrapolate` designates how the cell average is
 *computed.
 *  0  : Return stadnard cell average on iX
 * -1 : Extrapolate left, e.g.,  polynomial from iX+1 into iX
 * +1 : Extrapolate right, e.g.,  polynomial from iX-1 into iX
 **/
Real CellAverage( View3D<Real> U, const GridStructure *Grid,
                  const ModalBasis *Basis, const int iCF, const int iX,
                  const int extrapolate ) {
  const int nNodes = Grid->Get_nNodes( );

  Real avg  = 0.0;
  Real mass = 0.0;
  Real X;

  // Used to set loop bounds
  int mult  = 1;
  int end   = nNodes;
  int start = 0;

  if ( extrapolate == -1 ) mult = 1;
  if ( extrapolate == +0 ) mult = 0;
  if ( extrapolate == +1 ) mult = 2;

  if ( extrapolate == 0 ) {
    start = 0;
  } else {
    start = 2 + mult * nNodes;
  }
  end = start + nNodes - 1;

  for ( int iN = start; iN < end; iN++ ) {
    X = Grid->NodeCoordinate( iX + extrapolate, iN - start );
    mass += Grid->Get_Weights( iN - start ) * Grid->Get_SqrtGm( X ) *
            Grid->Get_Widths( iX + extrapolate );
    avg += Grid->Get_Weights( iN - start ) *
           Basis->basis_eval( U, iX + extrapolate, iCF, iN + 1 ) *
           Grid->Get_SqrtGm( X ) * Grid->Get_Widths( iX + extrapolate );
  }

  return avg / mass;
}

/**
 * Modify polynomials a la
 * H. Zhu et al 2020, simple and high-order
 * compact WENO RKDG slope limiter
 **/
void ModifyPolynomial( const View3D<Real> U, View2D<Real> modified_polynomial,
                       const Real gamma_i, const Real gamma_l,
                       const Real gamma_r, const int iX, const int iCQ ) {
  const Real Ubar_i = U( iCQ, iX, 0 );
  const Real fac    = 0.9;
  const int order   = U.extent( 2 );

  modified_polynomial( 0, 0 ) = Ubar_i;
  modified_polynomial( 2, 0 ) = Ubar_i;
  modified_polynomial( 0, 1 ) = fac * U( iCQ, iX - 1, 1 );
  modified_polynomial( 2, 1 ) = fac * U( iCQ, iX + 1, 1 );

  for ( int k = 2; k < order; k++ ) {
    modified_polynomial( 0, k ) = 0.0;
    modified_polynomial( 2, k ) = 0.0;
  }
  for ( int k = 0; k < order; k++ ) {
    modified_polynomial( 1, k ) =
        U( iCQ, iX, k ) / gamma_i -
        ( gamma_l / gamma_i ) * modified_polynomial( 0, k ) -
        ( gamma_r / gamma_i ) * modified_polynomial( 2, k );
  }
}

// WENO smoothness indicator beta
Real SmoothnessIndicator( const View3D<Real> U,
                          const View2D<Real> modified_polynomial,
                          const GridStructure *Grid, const int iX, const int i,
                          const int iCQ ) {
  const int k = U.extent( 2 );

  Real beta = 0.0; // output var
  for ( int s = 1; s < k; s++ ) { // loop over modes
    // integrate mode on cell
    Real local_sum = 0.0;
    for ( int iN = 0; iN < k; iN++ ) {
      auto X = Grid->NodeCoordinate( iX, iN );
      local_sum += Grid->Get_Weights( iN ) *
                   std::pow( modified_polynomial( i, s ) *
                                 ModalBasis::dLegendreN( k, s, X ),
                             2.0 ) *
                   std::pow( Grid->Get_Widths( iX ), 2.0 * s );
    }
    beta += local_sum;
  }
  return beta;
}

Real NonLinearWeight( const Real gamma, const Real beta, const Real tau,
                      const Real eps ) {
  return gamma * ( 1.0 + tau / ( eps + beta ) );
}

// weno-z tau variable
Real Tau( const Real beta_l, const Real beta_i, const Real beta_r,
          const Real weno_r ) {
  return std::pow(
      ( std::fabs( beta_i - beta_l ) + std::fabs( beta_i - beta_r ) ) / 2.0,
      weno_r );
}
