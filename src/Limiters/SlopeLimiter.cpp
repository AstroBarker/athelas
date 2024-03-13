/**
 * File     :  SlopeLimiter.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Classes for slope limters
 * Contains : SlopeLimiter
 * TODO: Restructure loops iCF...iX
 **/

#include <algorithm> /* std::min, std::max */
#include <cstdlib>   /* abs */
#include <limits>

#include "Kokkos_Core.hpp"

#include "CharacteristicDecomposition.hpp"
#include "Error.hpp"
#include "Grid.hpp"
#include "LinearAlgebraModules.hpp"
#include "PolynomialBasis.hpp"
#include "SlopeLimiter.hpp"
#include "SlopeLimiter_Utilities.hpp"
#include <iostream>

/**
 * The constructor creates the matrices structures for applying the slope
 * limiter
 **/
SlopeLimiter::SlopeLimiter( GridStructure *Grid, ProblemIn *pin )
    : order( pin->pOrder ),
      CharacteristicLimiting_Option( pin->Characteristic ),
      TCI_Option( pin->TCI_Option ), TCI_Threshold( pin->TCI_Threshold ),
      gamma_l( pin->gamma_l ), gamma_i( pin->gamma_i ), gamma_r( pin->gamma_r ),
      weno_r( pin->weno_r ),
      modified_polynomial( "modified_polynomial", 3, pin->pOrder ),
      R( "R Matrix", 3, 3, Grid->Get_nElements( ) + 2 * Grid->Get_Guard( ) ),
      R_inv( "invR Matrix", 3, 3,
             Grid->Get_nElements( ) + 2 * Grid->Get_Guard( ) ),
      U_c_T( "U_c_T", 3 ), w_c_T( "w_c_T", 3 ), Mult( "Mult", 3 ),
      D( "TCI", 3, Grid->Get_nElements( ) + 2 * Grid->Get_Guard( ) ),
      LimitedCell( "LimitedCell",
                   Grid->Get_nElements( ) + 2 * Grid->Get_Guard( ) ) {}

/**
 * Apply the Troubled Cell Indicator of Fu & Shu (2017)
 * to flag cells for limiting
 **/
void SlopeLimiter::DetectTroubledCells( View3D U, GridStructure *Grid,
                                        const ModalBasis *Basis ) {
  const int ilo = Grid->Get_ilo( );
  const int ihi = Grid->Get_ihi( );

  Real denominator = 0.0;

  // Cell averages by extrapolating L and R neighbors into current cell

  // TODO: Kokkos
  for ( int iCF = 0; iCF < 3; iCF++ )
    for ( int iX = ilo; iX <= ihi; iX++ ) {

      if ( iCF == 1 ) continue; /* skip velocit */

      Real result   = 0.0;
      Real cell_avg = U( iCF, iX, 0 );

      // Extrapolate neighboring poly representations into current cell
      // and compute the new cell averages
      Real cell_avg_L_T =
          CellAverage( U, Grid, Basis, iCF, iX + 1, -1 ); // from right
      Real cell_avg_R_T =
          CellAverage( U, Grid, Basis, iCF, iX - 1, +1 ); // from left
      Real cell_avg_L = U( iCF, iX - 1, 0 );              // native left
      Real cell_avg_R = U( iCF, iX + 1, 0 );              // native right

      result += ( std::abs( cell_avg - cell_avg_L_T ) +
                  std::abs( cell_avg - cell_avg_R_T ) );

      denominator =
          std::max( std::max( std::abs( cell_avg_L ), std::abs( cell_avg_R ) ),
                    cell_avg );

      D( iCF, iX ) = result / denominator;
    }
}

/**
 * Apply the slope limiter. We use a compact stencil WENO-Z limiter
 * H. Zhu 2020, simple, high-order compact WENO RKDG slope limiter
 **/
void SlopeLimiter::ApplySlopeLimiter( View3D U, GridStructure *Grid,
                                      const ModalBasis *Basis ) {

  // Do not apply for first order method. No slopes!
  if ( order == 1 ) {
    return;
  }

  const int &ilo  = Grid->Get_ilo( );
  const int &ihi  = Grid->Get_ihi( );
  const int nvars = U.extent( 0 );

  /* map to characteristic vars */
  if ( CharacteristicLimiting_Option ) {
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
  // Exit if we don't need to limit slopes
  // TODO: matter if we TCI on characteristics?
  if ( TCI_Option ) DetectTroubledCells( U, Grid, Basis );

  for ( int iCF = 0; iCF < nvars; iCF++ ) {
    for ( int iX = ilo; iX <= ihi; iX++ ) {

      this->LimitedCell( iX ) = 0;

      // Check if TCI val is less than TCI_Threshold
      int j = 0;
      for ( int iCF = 0; iCF < nvars; iCF++ ) {
        if ( this->D( iCF, iX ) > this->TCI_Threshold && this->TCI_Option ) {
          j++;
        }
      } // end check TCI

      if ( j != 0 || !TCI_Option ) {

        // modify polynomials
        ModifyPolynomial( U, iX, iCF );

        const Real beta_l =
            SmoothnessIndicator( U, Grid, iX, 0, iCF ); // iX - 1
        const Real beta_i = SmoothnessIndicator( U, Grid, iX, 1, iCF ); // iX
        const Real beta_r =
            SmoothnessIndicator( U, Grid, iX, 2, iCF ); // iX + 1
        const Real tau = Tau( beta_l, beta_i, beta_r );

        // nonlinear weights w
        const Real dx_i = Grid->Get_Widths( iX );
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
  if ( CharacteristicLimiting_Option ) {
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

/**
 * Return the cell average of a field iCF on cell iX.
 * The parameter `int extrapolate` designates how the cell average is
 *computed. 0  : Return stadnard cell average on iX -1 : Extrapolate
 *polynomial from iX+1 into iX +1 : Extrapolate polynomial from iX-1 into iX
 **/
Real SlopeLimiter::CellAverage( View3D U, GridStructure *Grid,
                                const ModalBasis *Basis, const int iCF,
                                const int iX, const int extrapolate ) const {
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
    X = Grid->NodeCoordinate( iX + extrapolate,
                              iN ); // Need the metric on target cell
    mass += Grid->Get_Weights( iN - start ) * Grid->Get_SqrtGm( X ) *
            Grid->Get_Widths(
                iX + extrapolate ); // / Basis.BasisEval( U,
                                    // iX+extrapolate, 0, iN+1, false );
    avg += Grid->Get_Weights( iN - start ) *
           Basis->BasisEval( U, iX + extrapolate, iCF, iN + 1 ) *
           Grid->Get_SqrtGm( X ) * Grid->Get_Widths( iX + extrapolate );
  }

  return avg / mass;
}

/**
 * Modify polynomials a la
 * H. Zhu et al 2020, simple and high-order
 * compact WENO RKDG slope limiter
 **/
void SlopeLimiter::ModifyPolynomial( const View3D U, const int iX,
                                     const int iCQ ) {
  const Real Ubar_i = U( iCQ, iX, 0 );
  const Real fac    = 1.0;

  modified_polynomial( 0, 0 ) = Ubar_i;
  modified_polynomial( 2, 0 ) = Ubar_i;
  modified_polynomial( 0, 1 ) = fac * U( iCQ, iX - 1, 1 );
  modified_polynomial( 2, 1 ) = fac * U( iCQ, iX + 1, 1 );

  for ( int k = 2; k < this->order; k++ ) {
    modified_polynomial( 0, k ) = 0.0;
    modified_polynomial( 2, k ) = 0.0;
  }
  for ( int k = 0; k < this->order; k++ ) {
    modified_polynomial( 1, k ) =
        U( iCQ, iX, k ) / gamma_i -
        ( gamma_l / gamma_i ) * modified_polynomial( 0, k ) -
        ( gamma_r / gamma_i ) * modified_polynomial( 2, k );
  }
}

// WENO smoothness indicator beta
Real SlopeLimiter::SmoothnessIndicator( const View3D U,
                                        const GridStructure *Grid, const int iX,
                                        const int i, const int iCQ ) const {
  const int k = U.extent( 2 ) - 1;

  Real beta = 0.0;                 // output var
  for ( int s = 1; s <= k; s++ ) { // loop over modes
    // integrate mode on cell
    Real local_sum = 0.0;
    for ( int iN = 0; iN < k; iN++ ) {
      local_sum += Grid->Get_Weights( iN ) *
                   std::pow( modified_polynomial( i, s ), 2.0 );
    }
    beta += local_sum;
  }
  return beta;
}

Real SlopeLimiter::NonLinearWeight( const Real gamma, const Real beta,
                                    const Real tau, const Real eps ) const {
  return gamma * ( 1.0 + tau / ( eps + beta ) );
}

Real SlopeLimiter::Tau( const Real beta_l, const Real beta_i,
                        const Real beta_r ) const {
  const Real r = this->weno_r;
  return std::pow(
      ( std::fabs( beta_i - beta_l ) + std::fabs( beta_i - beta_r ) ) / 2.0,
      r );
}

// LimitedCell accessor
int SlopeLimiter::Get_Limited( int iX ) const { return LimitedCell( iX ); }
