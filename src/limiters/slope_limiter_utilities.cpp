/**
 * @file slope_limiter_utilities.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Utility functions for slope limiters.
 */

#include <algorithm> // std::min, std::max
#include <cmath>
#include <cstdlib> /* abs */
#include <iostream>
#include <limits>

#include "slope_limiter.hpp"
#include "slope_limiter_utilities.hpp"
#include "utilities.hpp"

namespace limiter_utilities {

auto initialize_slope_limiter( const GridStructure* grid, const ProblemIn* pin,
                               const int nvars ) -> SlopeLimiter {
  SlopeLimiter S_Limiter;
  if ( utilities::to_lower( pin->limiter_type ) == "minmod" ) {
    S_Limiter = TVDMinmod( grid, pin, nvars );
  } else {
    S_Limiter = WENO( grid, pin, nvars );
  }
  return S_Limiter;
}

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
auto barth_jespersen( double U_v_L, double U_v_R, double U_c_L, double U_c_T,
                      double U_c_R, double alpha ) -> double {
  // Get U_min, U_max
  double U_min_L = 10000000.0 * U_c_T;
  double U_min_R = 10000000.0 * U_c_T;
  double U_max_L = std::numeric_limits<double>::epsilon( ) * U_c_T * 0.00001;
  double U_max_R = std::numeric_limits<double>::epsilon( ) * U_c_T * 0.00001;

  U_min_L = std::min( U_min_L, std::min( U_c_T, U_c_L ) );
  U_max_L = std::max( U_max_L, std::max( U_c_T, U_c_L ) );
  U_min_R = std::min( U_min_R, std::min( U_c_T, U_c_R ) );
  U_max_R = std::max( U_max_R, std::max( U_c_T, U_c_R ) );

  // loop of cell certices
  double phi_L = 0.0;
  double phi_R = 0.0;

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
void detect_troubled_cells( View3D<double> U, View2D<double> D,
                            const GridStructure* grid,
                            const ModalBasis* basis ) {
  const int nvars = U.extent( 0 );
  const int ilo   = grid->get_ilo( );
  const int ihi   = grid->get_ihi( );

  // Cell averages by extrapolating L and R neighbors into current cell

  for ( int iC = 0; iC < nvars; iC++ ) {
    if ( iC == 1 ) {
      continue; /* skip momenta */
    }
    Kokkos::parallel_for(
        "SlopeLimiter :: TCI", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
        KOKKOS_LAMBDA( const int iX ) {
          double denominator = 0.0;
          double result      = 0.0;
          double cell_avg    = U( iC, iX, 0 );

          // Extrapolate neighboring poly representations into current cell
          // and compute the new cell averages
          double cell_avg_L_T =
              cell_average( U, grid, basis, iC, iX + 1, -1 ); // from right
          double cell_avg_R_T =
              cell_average( U, grid, basis, iC, iX - 1, +1 ); // from left
          double cell_avg_L = U( iC, iX - 1, 0 ); // native left
          double cell_avg_R = U( iC, iX + 1, 0 ); // native right

          result += ( std::abs( cell_avg - cell_avg_L_T ) +
                      std::abs( cell_avg - cell_avg_R_T ) );

          denominator = std::max(
              std::max( std::abs( cell_avg_L ), std::abs( cell_avg_R ) ),
              cell_avg );

          D( iC, iX ) = result / denominator;
        } ); // par_for iX
  } // loop iC;
}

/**
 * Return the cell average of a field iCF on cell iX.
 * The parameter `int extrapolate` designates how the cell average is
 *computed.
 *  0  : Return standard cell average on iX
 * -1 : Extrapolate left, e.g.,  polynomial from iX+1 into iX
 * +1 : Extrapolate right, e.g.,  polynomial from iX-1 into iX
 **/
auto cell_average( View3D<double> U, const GridStructure* grid,
                   const ModalBasis* basis, const int iCF, const int iX,
                   const int extrapolate ) -> double {
  const int nNodes = grid->get_n_nodes( );

  double avg  = 0.0;
  double mass = 0.0;
  double X    = 0.0;

  // Used to set loop bounds
  int mult  = 1;
  int end   = nNodes;
  int start = 0;

  if ( extrapolate == -1 ) {
    mult = 1;
  }
  if ( extrapolate == +0 ) {
    mult = 0;
  }
  if ( extrapolate == +1 ) {
    mult = 2;
  }

  if ( extrapolate == 0 ) {
    start = 0;
  } else {
    start = 2 + mult * nNodes;
  }
  end = start + nNodes - 1;

  for ( int iN = start; iN < end; iN++ ) {
    X = grid->node_coordinate( iX + extrapolate, iN - start );
    mass += grid->get_weights( iN - start ) * grid->get_sqrt_gm( X ) *
            grid->get_widths( iX + extrapolate ); // TODO(astrobarker) rho
    avg += grid->get_weights( iN - start ) *
           basis->basis_eval( U, iX + extrapolate, iCF, iN + 1 ) *
           grid->get_sqrt_gm( X ) * grid->get_widths( iX + extrapolate );
  }

  return avg / mass;
}

/**
 * Modify polynomials a la
 * H. Zhu et al 2020, simple and high-order
 * compact WENO RKDG slope limiter
 **/
void modify_polynomial( const View3D<double> U, View2D<double> modified_polynomial,
                        const double gamma_i, const double gamma_l,
                        const double gamma_r, const int iX, const int iCQ ) {
  const double Ubar_i = U( iCQ, iX, 0 );
  const double fac    = 1.0;
  const int order   = U.extent( 2 );

  const double modified_p_slope_mag =
      fac *
      std::min( { U( iCQ, iX - 1, 1 ), U( iCQ, iX, 1 ), U( iCQ, iX + 1, 1 ) } );
  const int sign_l = utilities::SGN( U( iCQ, iX - 1, 1 ) );
  const int sign_r = utilities::SGN( U( iCQ, iX + 1, 1 ) );

  modified_polynomial( 0, 0 ) = Ubar_i;
  modified_polynomial( 2, 0 ) = Ubar_i;
  modified_polynomial( 0, 1 ) = sign_l * modified_p_slope_mag;
  modified_polynomial( 2, 1 ) = sign_r * modified_p_slope_mag;

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
auto smoothness_indicator( const View3D<double> U,
                           const View2D<double> modified_polynomial,
                           const GridStructure* grid, const ModalBasis* basis,
                           const int iX, const int i, const int /*iCQ*/ )
    -> double {
  const int k = U.extent( 2 );

  double beta = 0.0; // output var
  for ( int s = 1; s < k; s++ ) { // loop over modes
    // integrate mode on cell
    double local_sum = 0.0;
    for ( int iN = 0; iN < k; iN++ ) {
      auto X = grid->node_coordinate( iX, iN );
      local_sum += grid->get_weights( iN ) *
                   std::pow( modified_polynomial( i, s ) *
                                 ModalBasis::d_legendre_n( k, s, X ),
                             2.0 ) *
                   std::pow( grid->get_widths( iX ), 2.0 * s );
    }
    beta += local_sum;
  }
  return beta;
}

auto non_linear_weight( const double gamma, const double beta, const double tau,
                        const double eps ) -> double {
  return gamma * ( 1.0 + tau / ( eps + beta ) );
}

// weno-z tau variable
auto weno_tau( const double beta_l, const double beta_i, const double beta_r,
               const double weno_r ) -> double {
  return std::pow(
      ( std::abs( beta_i - beta_l ) + std::abs( beta_i - beta_r ) ) / 2.0,
      weno_r );
}

} // namespace limiter_utilities
