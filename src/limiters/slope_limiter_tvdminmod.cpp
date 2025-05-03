/**
 * @file slope_limiter_tvdminmod.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief TVB Minmod slope limiter for discontinuous Galerkin methods
 *
 * @details This file implements the Total Variation Diminishing (TVD) Minmod
 *          slope limiter based on the work of Cockburn & Shu. The limiter
 *          provides a robust, first-order accurate approach to preventing
 *          oscillations in discontinuous solutions.
 */

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

using namespace limiter_utilities;

/**
 * TVD Minmod limiter. See the Cockburn & Shu papers
 **/
void TVDMinmod::apply_slope_limiter( View3D<Real> U, const GridStructure* grid,
                                     const ModalBasis* basis ) {

  // Do not apply for first order method or if we don't want to.
  if ( order_ == 1 || !do_limiter_ ) {
    return;
  }

  constexpr static Real sl_threshold_ =
      1.0e-5; // TODO(astrobarker): move to input deck
  constexpr static Real EPS = 1.0e-10;

  const int& ilo  = grid->get_ilo( );
  const int& ihi  = grid->get_ihi( );
  const int nvars = U.extent( 0 );

  // --- Apply troubled cell indicator ---
  if ( tci_opt_ ) detect_troubled_cells( U, D_, grid, basis );

  // TODO(astrobarker): this is repeated code: clean up somehow
  /* map to characteristic vars */
  if ( characteristic_ ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: Minmod :: ToCharacteristic",
        Kokkos::RangePolicy<>( ilo, ihi + 1 ),
        KOKKOS_CLASS_LAMBDA( const int iX ) {
          // --- Characteristic Limiting Matrices ---
          // Note: using cell averages
          for ( int iC = 0; iC < nvars; iC++ ) {
            mult_( iC, iX ) = U( iC, iX, 0 );
          }

          auto R_i = Kokkos::subview( R_, Kokkos::ALL, Kokkos::ALL, iX );
          auto R_inv_i =
              Kokkos::subview( R_inv_, Kokkos::ALL, Kokkos::ALL, iX );
          auto U_c_T_i = Kokkos::subview( U_c_T_, Kokkos::ALL, iX );
          auto w_c_T_i = Kokkos::subview( w_c_T_, Kokkos::ALL, iX );
          auto Mult_i  = Kokkos::subview( mult_, Kokkos::ALL, iX );
          compute_characteristic_decomposition( Mult_i, R_i, R_inv_i );
            // store w_.. = invR @ U_..
            for ( int iC = 0; iC < nvars; iC++ ) {
              U_c_T_i( iC ) = U( iC, iX, 1 );
              w_c_T_i( iC ) = 0.0;
            }
            MAT_MUL( 1.0, R_inv_i, U_c_T_i, 1.0, w_c_T_i );

            for ( int iC = 0; iC < nvars; iC++ ) {
              U( iC, iX, 1 ) = w_c_T_i( iC );
            } // end loop vars
        } ); // par iX
  } // end map to characteristics

  for ( int iC = 0; iC < nvars; iC++ ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: Minmod", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
        KOKKOS_CLASS_LAMBDA( const int iX ) {
          limited_cell_( iX ) = 0;


          // Do nothing we don't need to limit slopes
          if ( (D_(0, iX) > tci_val_ || D_(2, iX) > tci_val_) || !tci_opt_ ) {

            /* Begin TVD Minmod Limiter */
            const Real s_i = U( iC, iX, 1 ); // target cell slope
            const Real c_i = U( iC, iX, 0 ); // target cell avg
            const Real c_p = U( iC, iX + 1, 0 ); // cell iX + 1 avg
            const Real c_m = U( iC, iX - 1, 0 ); // cell iX - 1 avg
            const Real dx  = grid->get_widths( iX );
            const Real new_slope =
                MINMOD_B( s_i, b_tvd_ * ( c_p - c_i ), b_tvd_ * ( c_i - c_m ),
                          dx, m_tvb_ );

            // check limited slope difference vs threshold
            if ( std::abs( new_slope - s_i ) > sl_threshold_ * std::max(std::abs(s_i), EPS) ) {
              // limit
              U( iC, iX, 1 ) = new_slope;
              if ( order_ > 2 && new_slope != s_i) { // remove any higher order_ contributions
                for ( int k = 2; k < order_; k++ ) {
                  U( iC, iX, k ) = 0.0;
                }
              }
            }
            /* End TVD Minmod Limiter */
            // The TVDMinmod part is really small... reusing a lot of code

            /* Note we have limited this cell */
            limited_cell_( iX ) = 1;

          } // end if "limit_this_cell"
        } ); // par_for iX
  } // end loop iC

  /* Map back to conserved variables */
  if ( characteristic_ ) {
    Kokkos::parallel_for(
        "SlopeLimiter :: Minmod :: FromCharacteristic",
        Kokkos::RangePolicy<>( ilo, ihi + 1 ),
        KOKKOS_CLASS_LAMBDA( const int iX ) {
          // --- Characteristic Limiting Matrices ---
          auto R_i = Kokkos::subview( R_, Kokkos::ALL, Kokkos::ALL, iX );
          auto U_c_T_i = Kokkos::subview( U_c_T_, Kokkos::ALL, iX );
          auto w_c_T_i = Kokkos::subview( w_c_T_, Kokkos::ALL, iX );
            // store U.. = R @ w..
            for ( int iC = 0; iC < nvars; iC++ ) {
              U_c_T_i( iC ) = U( iC, iX, 1 );
              w_c_T_i( iC ) = 0.0;
            }
            MAT_MUL( 1.0, R_i, U_c_T_i, 1.0, w_c_T_i );

            for ( int iC = 0; iC < nvars; iC++ ) {
              U( iC, iX, 1 ) = w_c_T_i( iC );
            } // end loop vars
        } ); // par_for iX
  } // end map from characteristics
} // end apply slope limiter

// limited_cell_ accessor
auto TVDMinmod::get_limited( const int iX ) const -> int {
  return limited_cell_( iX );
}
