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
  // const int nvars = U.extent( 0 );

  /* map to characteristic vars */
  if ( characteristic ) {
    for ( int iX = ilo; iX <= ihi; iX++ ) {
      // --- Characteristic Limiting Matrices ---
      // Note: using cell averages
      for ( int iCF = 0; iCF < nvars; iCF++ ) {
        Mult( iCF ) = U( iCF, iX, 0 );
      }

      auto R_i     = Kokkos::subview( R, Kokkos::ALL, Kokkos::ALL, iX );
      auto R_inv_i = Kokkos::subview( R_inv, Kokkos::ALL, Kokkos::ALL, iX );
      ComputeCharacteristicDecomposition( Mult, R_i, R_inv_i );
      for ( int k = 0; k < order; k++ ) {
        // store w_.. = invR @ U_..
        for ( int iCF = 0; iCF < nvars; iCF++ ) {
          U_c_T( iCF ) = U( iCF, iX, k );
          w_c_T( iCF ) = 0.0;
        }
        MatMul( 1.0, R_inv_i, U_c_T, 1.0, w_c_T );

        for ( int iCF = 0; iCF < nvars; iCF++ ) {
          U( iCF, iX, k ) = w_c_T( iCF );
        } // end loop vars
      } // end loop k
    } // end loop iX
  } // end map to characteristics

  // --- Apply troubled cell indicator ---
  // NOTE: applying TCI on characteristic vars
  // Could probably reduce some work by checking this first
  // and skipping as appropriate
  if ( tci_opt ) DetectTroubledCells( U, D, Grid, Basis );

  for ( int iCF = 0; iCF < nvars; iCF++ ) {
    for ( int iX = ilo; iX <= ihi; iX++ ) {

      this->LimitedCell( iX ) = 0;

      // Check if TCI val is less than TCI_Threshold
      int j = 0;
      if ( this->D( iCF, iX ) > this->tci_val && this->tci_opt ) {
        j++;
      }

      // Do nothing we don't need to limit slopes
      if ( j != 0 || !tci_opt ) {

        // modify polynomials
        ModifyPolynomial( U, modified_polynomial, gamma_i, gamma_l, gamma_r, iX,
                          iCF );

        const Real beta_l = SmoothnessIndicator( U, modified_polynomial, Grid,
                                                 iX, 0, iCF ); // iX - 1
        const Real beta_i = SmoothnessIndicator( U, modified_polynomial, Grid,
                                                 iX, 1, iCF ); // iX
        const Real beta_r = SmoothnessIndicator( U, modified_polynomial, Grid,
                                                 iX, 2, iCF ); // iX + 1
        const Real tau    = Tau( beta_l, beta_i, beta_r, weno_r );

        // nonlinear weights w
        const Real dx_i = 0.1 * Grid->Get_Widths( iX );
        Real w_l        = NonLinearWeight( this->gamma_l, beta_l, tau, dx_i );
        Real w_i        = NonLinearWeight( this->gamma_i, beta_i, tau, dx_i );
        Real w_r        = NonLinearWeight( this->gamma_r, beta_r, tau, dx_i );

        const Real sum_w = w_l + w_i + w_r;
        w_l /= sum_w;
        w_i /= sum_w;
        w_r /= sum_w;

        // update solution via WENO
        for ( int k = 0; k < this->order; k++ ) {
          U( iCF, iX, k ) = w_l * this->modified_polynomial( 0, k ) +
                            w_i * this->modified_polynomial( 1, k ) +
                            w_r * this->modified_polynomial( 2, k );
        }

        /* Note we have limited this cell */
        LimitedCell( iX ) = 1;

      } // end if "limit_this_cell"
    } // end loop iX
  } // end loop CF

  /* Map back to conserved variables */
  if ( characteristic ) {
    for ( int iX = ilo; iX <= ihi; iX++ ) {
      // --- Characteristic Limiting Matrices ---
      // Note: using cell averages
      for ( int iCF = 0; iCF < nvars; iCF++ ) {
        Mult( iCF ) = U( iCF, iX, 0 );
      }

      auto R_i     = Kokkos::subview( R, Kokkos::ALL, Kokkos::ALL, iX );
      auto R_inv_i = Kokkos::subview( R_inv, Kokkos::ALL, Kokkos::ALL, iX );
      for ( int k = 0; k < order; k++ ) {
        // store w_.. = invR @ U_..
        for ( int iCF = 0; iCF < nvars; iCF++ ) {
          U_c_T( iCF ) = U( iCF, iX, k );
          w_c_T( iCF ) = 0.0;
        }
        MatMul( 1.0, R_i, U_c_T, 1.0, w_c_T );

        for ( int iCF = 0; iCF < nvars; iCF++ ) {
          U( iCF, iX, k ) = w_c_T( iCF );
        } // end loop vars
      } // end loop k
    } // end loop iX
  } // end map from characteristics
} // end apply slope limiter

// LimitedCell accessor
int WENO::Get_Limited( const int iX ) const { return LimitedCell( iX ); }
#endif // SLOPE_LIMITER_WENO_HPP_
