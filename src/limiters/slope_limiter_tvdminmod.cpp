#ifndef SLOPE_LIMITER_TVDMINMOD_HPP_
#define SLOPE_LIMITER_TVDMINMOD_HPP_

/**
 * File     :  slope_limiter_tvdminmod.hpp
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
 * TVD Minmod limiter. See the Cockburn & Shu papers
 **/
void TVDMinmod::ApplySlopeLimiter( View3D<Real> U, const GridStructure *Grid,
                                   const ModalBasis *Basis ) {

  // Do not apply for first order method or if we don't want to.
  if ( order == 1 || !do_limiter ) {
    return;
  }

  const int &ilo = Grid->Get_ilo( );
  const int &ihi = Grid->Get_ihi( );
  // const int nvars = U.extent( 0 );

  // TODO: this is repeated code: clean up somehow
  /* map to characteristic vars */
  if ( characteristic ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: Minmod :: ToCharacteristic",
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
        "SlopeLimiter :: Minmod", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
        KOKKOS_LAMBDA( const int iX ) {
          this->LimitedCell( iX ) = 0;

          // Check if TCI val is less than TCI_Threshold
          int j = 0;
          if ( this->D( iC, iX ) > this->tci_val && this->tci_opt ) {
            j++;
          }

          // Do nothing we don't need to limit slopes
          if ( j != 0 || !tci_opt ) {

            /* Begin TVD Minmod Limiter */
            const Real sl_threshold = 1.0e-6; // TODO: move to input deck
            const Real s_i          = U( iC, iX, 1 ); // target cell slope
            const Real c_i          = U( iC, iX, 0 ); // target cell avg
            const Real c_p          = U( iC, iX + 1, 0 ); // cell iX + 1 avg
            const Real c_m          = U( iC, iX - 1, 0 ); // cell iX - 1 avg
            const Real dx           = Grid->Get_Widths( iX );
            const Real new_slope    = minmodB( s_i, b_tvd * ( c_p - c_i ),
                                               b_tvd * ( c_i - c_m ), dx, m_tvb );

            // check limited slope difference vs threshold
            if ( std::abs( new_slope - s_i ) > sl_threshold * s_i ) {
              // limit
              U( iC, iX, 1 ) = new_slope;
              if ( order > 2 ) { // remove any higher order contributions
                for ( int k = 2; k < order; k++ ) {
                  U( iC, iX, k ) = 0.0;
                }
              }
            }
            /* End TVD Minmod Limiter */
            // The TVDMinmod part is really small... reusing a lot of code

            /* Note we have limited this cell */
            LimitedCell( iX ) = 1;

          } // end if "limit_this_cell"
        } ); // par_for iX
  } // end loop iC

  /* Map back to conserved variables */
  if ( characteristic ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: Minmod :: FromCharacteristic",
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
int TVDMinmod::Get_Limited( const int iX ) const { return LimitedCell( iX ); }
#endif // SLOPE_LIMITER_TVDMINMOD_HPP_
