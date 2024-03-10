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
      modified_polynomial( "mmodified_polynomial", 3, pin->pOrder ),
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
 * Apply the slope limiter. We use a compact stencil WENO limiter
 * X. Zhong and C.-W. Shu 13, simple compact WENO RKDG slope limiter
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

  // --- Apply troubled cell indicator ---
  // Exit if we don't need to limit slopes

  // map to characteristic vars
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
        }
      }
    }
  }

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
      }

      if ( j != 0 || !TCI_Option ) {

        const Real beta_l = SmoothnessIndicator( U, Grid, iX - 1, iCF );
        const Real beta_i = SmoothnessIndicator( U, Grid, iX, iCF );
        const Real beta_r = SmoothnessIndicator( U, Grid, iX + 1, iCF );

        // nonlinear weights w
        Real w_l         = NonLinearWeight( this->gamma_l, beta_l );
        Real w_i         = NonLinearWeight( this->gamma_i, beta_i );
        Real w_r         = NonLinearWeight( this->gamma_r, beta_r );
        const Real sum_w = w_l + w_i + w_r;
        w_l /= sum_w;
        w_i /= sum_w;
        w_r /= sum_w;

        // modify polynomials
        ModifyPolynomial( U, iX, iCF );

        // update solution via WENO
        for ( int k = 0; k < this->order; k++ ) {
          U( iCF, iX, k ) = w_l * this->modified_polynomial( 0, k ) +
                            w_i * this->modified_polynomial( 1, k ) +
                            w_r * this->modified_polynomial( 2, k );
        }
      }

      /* Note we have limited this cell */
      // LimitedCell( iX ) = 1;
    }
  }

  if ( CharacteristicLimiting_Option ) {
    for ( int iX = ilo; iX <= ihi; iX++ ) {
      // --- Characteristic Limiting Matrices ---
      // Note: using cell averages
      for ( int iCF = 0; iCF < nvars; iCF++ ) {
        Mult( iCF ) = U( iCF, iX, 0 );
      }

      auto R_i     = Kokkos::subview( R, Kokkos::ALL, Kokkos::ALL, iX );
      auto R_inv_i = Kokkos::subview( R_inv, Kokkos::ALL, Kokkos::ALL, iX );
      // ComputeCharacteristicDecomposition( Mult, R_i, R_inv_i );
      for ( int k = 0; k < order; k++ ) {
        // store w_.. = invR @ U_..
        for ( int iCF = 0; iCF < nvars; iCF++ ) {
          U_c_T( iCF ) = U( iCF, iX, k );
          w_c_T( iCF ) = 0.0;
        }
        MatMul( 1.0, R_i, U_c_T, 1.0, w_c_T );

        for ( int iCF = 0; iCF < nvars; iCF++ ) {
          U( iCF, iX, k ) = w_c_T( iCF );
        }
      }
    }
  }
}

/**
 * Return the cell average of a field iCF on cell iX.
 * The parameter `int extrapolate` designates how the cell average is
 *computed. 0  : Return stadnard cell average on iX -1 : Extrapolate
 *polynomial from iX+1 into iX +1 : Extrapolate polynomial from iX-1 into iX
 **/
Real SlopeLimiter::CellAverage( View3D U, GridStructure *Grid,
                                const ModalBasis *Basis, const int iCF,
                                const int iX, const int extrapolate ) {
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
           Basis->BasisEval( U, iX + extrapolate, iCF, iN + 1, false ) *
           Grid->Get_SqrtGm( X ) * Grid->Get_Widths( iX + extrapolate );
  }

  return avg / mass;
}

/**
 * Modify polynomials a la
 * X. Zhong and C.-W. Shu 13, simple compact WENO RKDG slope limiter
 * UNUSED, point based version.
 **/
Real SlopeLimiter::ModifyPolynomial( const View3D U, const ModalBasis *Basis,
                                     const Real Ubar_i, const int iX,
                                     const int iCQ, const int iN ) {
  return Basis->BasisEval( U, iX, iCQ, iN, false ) - Ubar_i + U( iCQ, iX, 0 );
}

/**
 * Modify polynomials a la
 * X. Zhong and C.-W. Shu 13, simple compact WENO RKDG slope limiter
 **/
void SlopeLimiter::ModifyPolynomial( const View3D U, const int iX,
                                     const int iCQ ) {
  const Real Ubar_i = U( iCQ, iX, 0 );
  // set to target cell average
  modified_polynomial( 0, 0 ) = Ubar_i; // left
  modified_polynomial( 1, 0 ) = Ubar_i; // target cell
  modified_polynomial( 2, 0 ) = Ubar_i; // right
  for ( int k = 1; k < this->order; k++ ) {
    modified_polynomial( 0, k ) = U( iCQ, iX - 1, k );
    modified_polynomial( 1, k ) = U( iCQ, iX, k );
    modified_polynomial( 2, k ) = U( iCQ, iX + 1, k );
  }
}

// WENO smoothness indicator beta
Real SlopeLimiter::SmoothnessIndicator( const View3D U,
                                        const GridStructure *Grid, const int iX,
                                        const int iCQ ) {
  const int k   = U.extent( 2 ) - 1;
  const Real dx = Grid->Get_Widths( iX );

  Real beta = 0.0;                 // output var
  for ( int s = 1; s <= k; s++ ) { // loop over modes
    // integrate mode on cell
    Real local_sum = 0.0;
    for ( int iN = 0; iN < k; iN++ ) {
      // Q: how to evaluate derivaitvs at iN?
      local_sum += Grid->Get_Weights( iN ) * std::pow( U( iCQ, iX, s ), 2.0 );
    }
    local_sum *= std::pow( dx, 2.0 * s );
    beta += local_sum;
  }
  return beta;
}

Real SlopeLimiter::NonLinearWeight( const Real gamma, const Real beta ) {
  const Real eps = std::numeric_limits<Real>::epsilon( );
  const Real r   = this->weno_r;

  return gamma / std::pow( beta + eps, r );
}

// LimitedCell accessor
int SlopeLimiter::Get_Limited( int iX ) const { return LimitedCell( iX ); }
