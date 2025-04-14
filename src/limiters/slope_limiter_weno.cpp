/**
 * @file slope_limiter_weno.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Implementation of the WENO-Z slope limiter for discontinuous Galerkin
 *        methods
 *
 * @details This file implements the WENO-Z slope limiter based on H. Zhu 2020,
 *          "Simple, high-order compact WENO RKDG slope limiter". The limiter
 *          uses a compact stencil approach to maintain high-order accuracy
 * while preventing oscillations.
 */

#include <algorithm> /* std::min, std::max */
#include <cmath>
#include <cstdlib> /* abs */
#include <iostream>
#include <limits>

#include "Kokkos_Core.hpp"

#include "characteristic_decomposition.hpp"
#include "error.hpp"
#include "grid.hpp"
#include "linear_algebra.hpp"
#include "polynomial_basis.hpp"
#include "slope_limiter.hpp"
#include "slope_limiter_base.hpp"
#include "slope_limiter_utilities.hpp"

using namespace limiter_utilities;

/**
 * Apply the slope limiter. We use a compact stencil WENO-Z limiter
 * H. Zhu 2020, simple, high-order compact WENO RKDG slope limiter
 **/
void WENO::apply_slope_limiter( View3D<Real> U, const GridStructure* grid,
                                const ModalBasis* basis ) {

  // Do not apply for first order method or if we don't want to.
  if ( order_ == 1 || !do_limiter_ ) {
    return;
  }

  const int& ilo  = grid->get_ilo( );
  const int& ihi  = grid->get_ihi( );
  const int nvars = U.extent( 0 );

  /* map to characteristic vars */
  if ( characteristic_ ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: WENO :: ToCharacteristic",
        Kokkos::RangePolicy<>( ilo, ihi + 1 ),
        KOKKOS_CLASS_LAMBDA( const int iX ) {
          // --- Characteristic Limiting Matrices ---
          // Note: using cell averages
          for ( int iC = 0; iC < nvars_; iC++ ) {
            mult_( iC, iX ) = U( iC, iX, 0 );
          }

          auto R_i = Kokkos::subview( R_, Kokkos::ALL, Kokkos::ALL, iX );
          auto R_inv_i =
              Kokkos::subview( R_inv_, Kokkos::ALL, Kokkos::ALL, iX );
          auto U_c_T_i = Kokkos::subview( U_c_T_, Kokkos::ALL, iX );
          auto w_c_T_i = Kokkos::subview( w_c_T_, Kokkos::ALL, iX );
          auto mult_i  = Kokkos::subview( mult_, Kokkos::ALL, iX );
          compute_characteristic_decomposition( mult_i, R_i, R_inv_i );
          for ( int k = 0; k < order_; k++ ) {
            // store w_.. = invR @ U_..
            for ( int iC = 0; iC < nvars_; iC++ ) {
              U_c_T_i( iC ) = U( iC, iX, k );
              w_c_T_i( iC ) = 0.0;
            }
            MAT_MUL( 1.0, R_inv_i, U_c_T_i, 1.0, w_c_T_i );

            for ( int iC = 0; iC < nvars_; iC++ ) {
              U( iC, iX, k ) = w_c_T_i( iC );
            } // end loop vars
          } // end loop k
        } ); // par iX
  } // end map to characteristics

  // --- Apply troubled cell indicator ---
  // NOTE: applying TCI on characteristic vars
  // Could probably reduce some work by checking this first
  // and skipping as appropriate
  if ( tci_opt_ ) detect_troubled_cells( U, D_, grid, basis );

  for ( int iC = 0; iC < nvars_; iC++ ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: WENO", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
        KOKKOS_CLASS_LAMBDA( const int iX ) {
          limited_cell_( iX ) = 0;

          // Check if TCI val is less than TCI_Threshold
          int j = 0;
          if ( D_( iC, iX ) > tci_val_ && tci_opt_ ) {
            j++;
          }

          // Do nothing we don't need to limit slopes
          if ( j != 0 || !tci_opt_ ) {
            // get scratch modified_polynomial view for this cell's work
            auto modified_polynomial_i = Kokkos::subview(
                modified_polynomial_, iX, Kokkos::ALL, Kokkos::ALL );

            // modify polynomials
            modify_polynomial( U, modified_polynomial_i, gamma_i_, gamma_l_,
                               gamma_r_, iX, iC );

            const Real beta_l = smoothness_indicator(
                U, modified_polynomial_i, grid, iX, 0, iC ); // iX - 1
            const Real beta_i = smoothness_indicator( U, modified_polynomial_i,
                                                      grid, iX, 1, iC ); // iX
            const Real beta_r = smoothness_indicator(
                U, modified_polynomial_i, grid, iX, 2, iC ); // iX + 1
            const Real tau = weno_tau( beta_l, beta_i, beta_r, weno_r_ );

            // nonlinear weights w
            const Real dx_i = 0.1 * grid->get_widths( iX );
            Real w_l        = non_linear_weight( gamma_l_, beta_l, tau, dx_i );
            Real w_i        = non_linear_weight( gamma_i_, beta_i, tau, dx_i );
            Real w_r        = non_linear_weight( gamma_r_, beta_r, tau, dx_i );

            const Real sum_w = w_l + w_i + w_r;
            w_l /= sum_w;
            w_i /= sum_w;
            w_r /= sum_w;

            // update solution via WENO
            for ( int k = 0; k < order_; k++ ) {
              U( iC, iX, k ) = w_l * modified_polynomial_i( 0, k ) +
                               w_i * modified_polynomial_i( 1, k ) +
                               w_r * modified_polynomial_i( 2, k );
            }

            /* Note we have limited this cell */
            limited_cell_( iX ) = 1;

          } // end if "limit_this_cell"
        } ); // par_for iX
  } // end loop iC

  /* Map back to conserved variables */
  if ( characteristic_ ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: WENO :: FromCharacteristic",
        Kokkos::RangePolicy<>( ilo, ihi + 1 ),
        KOKKOS_CLASS_LAMBDA( const int iX ) {
          // --- Characteristic Limiting Matrices ---
          // Note: using cell averages
          for ( int iC = 0; iC < nvars_; iC++ ) {
            mult_( iC, iX ) = U( iC, iX, 0 );
          }

          auto R_i = Kokkos::subview( R_, Kokkos::ALL, Kokkos::ALL, iX );
          auto R_inv_i =
              Kokkos::subview( R_inv_, Kokkos::ALL, Kokkos::ALL, iX );
          auto U_c_T_i = Kokkos::subview( U_c_T_, Kokkos::ALL, iX );
          auto w_c_T_i = Kokkos::subview( w_c_T_, Kokkos::ALL, iX );
          auto mult_i  = Kokkos::subview( mult_, Kokkos::ALL, iX );
          for ( int k = 0; k < order_; k++ ) {
            // store w_.. = invR @ U_..
            for ( int iC = 0; iC < nvars_; iC++ ) {
              U_c_T_i( iC ) = U( iC, iX, k );
              w_c_T_i( iC ) = 0.0;
            }
            MAT_MUL( 1.0, R_i, U_c_T_i, 1.0, w_c_T_i );

            for ( int iC = 0; iC < nvars_; iC++ ) {
              U( iC, iX, k ) = w_c_T_i( iC );
            } // end loop vars
          } // end loop k
        } ); // par_for iX
  } // end map from characteristics
} // end apply slope limiter

// LimitedCell accessor
auto WENO::get_limited( const int iX ) const -> int {
  return limited_cell_( iX );
}
