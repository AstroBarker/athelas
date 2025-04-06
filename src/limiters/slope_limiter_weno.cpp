#ifndef SLOPE_LIMITER_WENO_HPP_
#define SLOPE_LIMITER_WENO_HPP_

/**
 * File     :  slope_limiter_weno.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Classes for slope limters
 * Contains : SlopeLimiter
 **/

#include <algorithm> /* std::min, std::max */
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

/**
 * Apply the slope limiter. We use a compact stencil WENO-Z limiter
 * H. Zhu 2020, simple, high-order compact WENO RKDG slope limiter
 **/
void WENO::ApplySlopeLimiter( View3D<Real> U, const GridStructure *Grid,
                              const ModalBasis *Basis ) {

  // Do not apply for first order method or if we don't want to.
  if ( order == 1 || !do_limiter ) {
    return;
  }

  const int &ilo = Grid->Get_ilo( );
  const int &ihi = Grid->Get_ihi( );
  const int nvars = U.extent( 0 );

  /* map to characteristic vars */
  if ( characteristic ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: WENO :: ToCharacteristic",
        Kokkos::RangePolicy<>( ilo, ihi + 1 ), KOKKOS_LAMBDA( const int iX ) {
          // --- Characteristic Limiting Matrices ---
          // Note: using cell averages
          for ( int iC = 0; iC < nvars; iC++ ) {
            Mult( iC, iX ) = U( iC, iX, 0 );
          }

          auto R_i     = Kokkos::subview( R, Kokkos::ALL, Kokkos::ALL, iX );
          auto R_inv_i = Kokkos::subview( R_inv, Kokkos::ALL, Kokkos::ALL, iX );
          auto U_c_T_i = Kokkos::subview( U_c_T, Kokkos::ALL, iX );
          auto w_c_T_i = Kokkos::subview( w_c_T, Kokkos::ALL, iX );
          auto Mult_i  = Kokkos::subview( Mult, Kokkos::ALL, iX );
          ComputeCharacteristicDecomposition( Mult_i, R_i, R_inv_i );
          for ( int k = 0; k < order; k++ ) {
            // store w_.. = invR @ U_..
            for ( int iC = 0; iC < nvars; iC++ ) {
              U_c_T_i( iC ) = U( iC, iX, k );
              w_c_T_i( iC ) = 0.0;
            }
            MatMul( 1.0, R_inv_i, U_c_T_i, 1.0, w_c_T_i );

            for ( int iC = 0; iC < nvars; iC++ ) {
              U( iC, iX, k ) = w_c_T_i( iC );
            } // end loop vars
          } // end loop k
        } ); // par iX
  } // end map to characteristics

  // --- Apply troubled cell indicator ---
  // NOTE: applying TCI on characteristic vars
  // Could probably reduce some work by checking this first
  // and skipping as appropriate
  if ( tci_opt ) DetectTroubledCells( U, D, Grid, Basis );

  for ( int iC = 0; iC < nvars; iC++ ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: WENO", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
        KOKKOS_LAMBDA( const int iX ) {
          this->LimitedCell( iX ) = 0;

          // Check if TCI val is less than TCI_Threshold
          int j = 0;
          if ( this->D( iC, iX ) > this->tci_val && this->tci_opt ) {
            j++;
          }

          // Do nothing we don't need to limit slopes
          if ( j != 0 || !tci_opt ) {
            // get scratch modified_polynomial view for this cell's work
            auto modified_polynomial_i = Kokkos::subview(
                modified_polynomial, iX, Kokkos::ALL, Kokkos::ALL );

            // modify polynomials
            ModifyPolynomial( U, modified_polynomial_i, gamma_i, gamma_l,
                              gamma_r, iX, iC );

            const Real beta_l = SmoothnessIndicator(
                U, modified_polynomial_i, Grid, iX, 0, iC ); // iX - 1
            const Real beta_i = SmoothnessIndicator( U, modified_polynomial_i,
                                                     Grid, iX, 1, iC ); // iX
            const Real beta_r = SmoothnessIndicator(
                U, modified_polynomial_i, Grid, iX, 2, iC ); // iX + 1
            const Real tau = Tau( beta_l, beta_i, beta_r, weno_r );

            // nonlinear weights w
            const Real dx_i = 0.1 * Grid->Get_Widths( iX );
            Real w_l = NonLinearWeight( this->gamma_l, beta_l, tau, dx_i );
            Real w_i = NonLinearWeight( this->gamma_i, beta_i, tau, dx_i );
            Real w_r = NonLinearWeight( this->gamma_r, beta_r, tau, dx_i );

            const Real sum_w = w_l + w_i + w_r;
            w_l /= sum_w;
            w_i /= sum_w;
            w_r /= sum_w;

            // update solution via WENO
            for ( int k = 0; k < this->order; k++ ) {
              U( iC, iX, k ) = w_l * modified_polynomial_i( 0, k ) +
                               w_i * modified_polynomial_i( 1, k ) +
                               w_r * modified_polynomial_i( 2, k );
            }

            /* Note we have limited this cell */
            LimitedCell( iX ) = 1;

          } // end if "limit_this_cell"
        } ); // par_for iX
  } // end loop iC

  /* Map back to conserved variables */
  if ( characteristic ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: WENO :: FromCharacteristic",
        Kokkos::RangePolicy<>( ilo, ihi + 1 ), KOKKOS_LAMBDA( const int iX ) {
          // --- Characteristic Limiting Matrices ---
          // Note: using cell averages
          for ( int iC = 0; iC < nvars; iC++ ) {
            Mult( iC, iX ) = U( iC, iX, 0 );
          }

          auto R_i     = Kokkos::subview( R, Kokkos::ALL, Kokkos::ALL, iX );
          auto R_inv_i = Kokkos::subview( R_inv, Kokkos::ALL, Kokkos::ALL, iX );
          auto U_c_T_i = Kokkos::subview( U_c_T, Kokkos::ALL, iX );
          auto w_c_T_i = Kokkos::subview( w_c_T, Kokkos::ALL, iX );
          auto Mult_i  = Kokkos::subview( Mult, Kokkos::ALL, iX );
          for ( int k = 0; k < order; k++ ) {
            // store w_.. = invR @ U_..
            for ( int iC = 0; iC < nvars; iC++ ) {
              U_c_T_i( iC ) = U( iC, iX, k );
              w_c_T_i( iC ) = 0.0;
            }
            MatMul( 1.0, R_i, U_c_T_i, 1.0, w_c_T_i );

            for ( int iC = 0; iC < nvars; iC++ ) {
              U( iC, iX, k ) = w_c_T_i( iC );
            } // end loop vars
          } // end loop k
        } ); // par_for iX
  } // end map from characteristics
} // end apply slope limiter

// LimitedCell accessor
int WENO::Get_Limited( const int iX ) const { return LimitedCell( iX ); }
#endif // SLOPE_LIMITER_WENO_HPP_
