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

#include "Kokkos_Core.hpp"

#include "CharacteristicDecomposition.h"
#include "Error.h"
#include "Grid.h"
#include "LinearAlgebraModules.h"
#include "PolynomialBasis.h"
#include "SlopeLimiter.h"
#include "SlopeLimiter_Utilities.h"
#include <iostream>

/**
 * The constructor creates the matrices structures for applying the slope
 *limiter
 **/
SlopeLimiter::SlopeLimiter( GridStructure *Grid, UInt pOrder,
                            Real SlopeLimiterThreshold, Real alpha_val,
                            bool CharacteristicLimitingOption, bool TCIOption,
                            Real TCI_Threshold_val )
    : order( pOrder ), SlopeLimiter_Threshold( SlopeLimiterThreshold ),
      alpha( alpha_val ),
      CharacteristicLimiting_Option( CharacteristicLimitingOption ),
      TCI_Option( TCIOption ), TCI_Threshold( TCI_Threshold_val ),
      R( "R Matrix" ), R_inv( "invR Matrix" ), SlopeDifference( "SlopeDiff" ),
      dU( "dU" ), d2U( "d2U" ), d2w( "d2w" ), U_c_L( "U_c_L" ),
      U_c_T( "U_c_T" ), U_c_R( "U_c_R" ), U_v_L( "U_v_L" ), U_v_R( "U_v_R" ),
      dU_c_L( "dU_c_L" ), dU_c_T( "dU_c_T" ), dU_c_R( "dU_c_R" ),
      dU_v_L( "dU_v_L" ), dU_v_R( "dU_v_R" ), w_c_L( "w_c_L" ),
      w_c_T( "w_c_T" ), w_c_R( "w_c_R" ), w_v_L( "w_v_L" ), w_v_R( "U_v_R" ),
      dw_c_L( "dw_c_L" ), dw_c_T( "dw_c_T" ), dw_c_R( "dw_c_R" ),
      dw_v_L( "dw_v_L" ), dw_v_R( "dw_v_R" ), Mult1( "Mult1" ),
      Mult2( "Mult2" ), Mult3( "Mult3" ),
      D( "TCI", 3, Grid->Get_nElements( ) + 2 * Grid->Get_Guard( ) ),
      LimitedCell( "LimitedCell",
                   Grid->Get_nElements( ) + 2 * Grid->Get_Guard( ) )
{
}

/**
 * Apply the Troubled Cell Indicator of Fu & Shu (2017)
 * to flag cells for limiting
 **/
void SlopeLimiter::DetectTroubledCells( Kokkos::View<Real***> U,
                                        GridStructure *Grid,
                                        ModalBasis *Basis )
{
  const UInt ilo = Grid->Get_ilo( );
  const UInt ihi = Grid->Get_ihi( );

  Real denominator = 0.0;

  // Cell averages by extrapolating L and R neighbors into current cell

  // TODO: Kokkos
  for ( UInt iCF = 0; iCF < 3; iCF++ )
    for ( UInt iX = ilo; iX <= ihi; iX++ )
    {

      if ( iCF == 1 ) continue; /* skip velocit */

      Real result   = 0.0;
      Real cell_avg = U( iCF, iX, 0 );

      // Extrapolate neighboring poly representations into current cell
      // and compute the new cell averages
      Real cell_avg_L_T =
          CellAverage( U, Grid, Basis, iCF, iX + 1, -1 ); // from right
      Real cell_avg_R_T =
          CellAverage( U, Grid, Basis, iCF, iX - 1, +1 ); // from left
      Real cell_avg_L = U( iCF, iX - 1, 0 );            // native left
      Real cell_avg_R = U( iCF, iX + 1, 0 );            // native right

      result += ( std::abs( cell_avg - cell_avg_L_T ) +
                  std::abs( cell_avg - cell_avg_R_T ) );

      denominator =
          std::max( std::max( std::abs( cell_avg_L ), std::abs( cell_avg_R ) ),
                    cell_avg );

      D( iCF, iX ) = result / denominator;
    }
}

/**
 * Apply the slope limiter. We use a vertex based, heirarchical slope limiter.
 **/
void SlopeLimiter::ApplySlopeLimiter( Kokkos::View<Real***> U,
                                      GridStructure *Grid,
                                      ModalBasis *Basis )
{

  // Do not apply for first order method. No slopes!
  if ( order == 1 )
  {
    return;
  }

  const UInt& ilo    = Grid->Get_ilo( );
  const UInt& ihi    = Grid->Get_ihi( );
  const UInt& nNodes = Grid->Get_nNodes( );

  // --- Apply troubled cell indicator ---
  // Exit if we don't need to limit slopes

  if ( TCI_Option ) DetectTroubledCells( U, Grid, Basis );

  for ( UInt iX = ilo; iX <= ihi; iX++ )
  {

    LimitedCell( iX ) = 0;

    // Check if TCI val is less than TCI_Threshold
    int j = 0;
    for ( UInt iCF = 0; iCF < 3; iCF++ )
    {
      if ( D( iCF, iX ) > TCI_Threshold && TCI_Option )
      {
        j++;
      }
    }

    if ( j == 0 && TCI_Option ) continue;

    /* Note we have limited this cell */
    LimitedCell( iX ) = 1;

    for ( int i = 0; i < 3; i++ )
    {
      d2w( i )   = 0.0;
      Mult1( i ) = 0.0;
      Mult2( i ) = 0.0;
      Mult3( i ) = 0.0;
    }

    // --- Characteristic Limiting Matrices ---
    // Note: using cell averages

    if ( CharacteristicLimiting_Option )
    {
      for ( int iCF = 0; iCF < 3; iCF++ )
      {
        Mult2( iCF ) = U( iCF, iX, 0 );
      }
      ComputeCharacteristicDecomposition( Mult2, R, R_inv );
    }
    else
    {
      IdentityMatrix( R, 3 );
      IdentityMatrix( R_inv, 3 );
    }

    // ! Anything needed for boundaries? !

    // --- Limit Quadratic Term ---
    if ( order >= 3 )
    {
      for ( int iCF = 0; iCF < 3; iCF++ )
      {
        Mult1( iCF ) = U( iCF, iX, 2 );
      }
      MatMul( 1.0, R_inv, Mult1, 1.0, d2w );

      LimitQuadratic( U, Basis, d2w, iX, nNodes );
    }

    // --- Compute info for limiter ---
    for ( UInt iCF = 0; iCF < 3; iCF++ )
    {
      Mult1( iCF ) = 0.0;
      Mult2( iCF ) = 0.0;
      Mult3( iCF ) = 0.0;

      U_c_L( iCF ) = U( iCF, iX - 1, 0 );
      U_c_T( iCF ) = U( iCF, iX, 0 );
      U_c_R( iCF ) = U( iCF, iX + 1, 0 );

      dU_c_T( iCF ) = U( iCF, iX, 1 );

      U_v_L( iCF ) = Basis->BasisEval( U, iX, iCF, 0, false );
      U_v_R( iCF ) = Basis->BasisEval( U, iX, iCF, nNodes + 1, false );

      // initialize characteristic forms
      w_c_L( iCF ) = 0.0;
      w_c_T( iCF ) = 0.0;
      w_c_R( iCF ) = 0.0;

      w_v_L( iCF ) = 0.0;
      w_v_R( iCF ) = 0.0;

      dw_c_T( iCF ) = 0.0;
    }

    // --- Map limiter variables to characteristics ---

    // store w_.. = invR @ U_..
    MatMul( 1.0, R_inv, U_c_L, 1.0, w_c_L );
    MatMul( 1.0, R_inv, U_c_T, 1.0, w_c_T );
    MatMul( 1.0, R_inv, U_c_R, 1.0, w_c_R );

    MatMul( 1.0, R_inv, U_v_L, 1.0, w_v_L );
    MatMul( 1.0, R_inv, U_v_R, 1.0, w_v_R );

    MatMul( 1.0, R_inv, dU_c_T, 1.0, dw_c_T );

    // Limited Slopes
    for ( UInt iCF = 0; iCF < 3; iCF++ )
    {
      Phi1 = BarthJespersen( w_v_L( iCF ), w_v_R( iCF ), w_c_L( iCF ),
                             w_c_T( iCF ), w_c_R( iCF ), alpha );

      dU( iCF ) = Phi1 * dw_c_T( iCF ); // Multiply slope by Phi1
      if ( order >= 3 ) d2U( iCF ) = Phi1 * d2w( iCF ); // 2nd derivative
    }

    // Transform back to conserved quantities
    if ( CharacteristicLimiting_Option )
    {
      // dU -> R dU
      MatMul( 1.0, R, dU, 1.0, Mult1 );
      // d2U -> R d2U
      if ( order >= 3 )
      {
        MatMul( 1.0, R, d2U, 1.0, Mult2 );
      }
      for ( UInt iCF = 0; iCF < 3; iCF++ )
      {
        dU( iCF ) = Mult1( iCF );
        if ( order >= 3 ) d2U( iCF ) = Mult2( iCF );
      }
    }

    // --- Compare Limited to Original Slopes ---
    for ( UInt iCF = 0; iCF < 3; iCF++ )
    {
      SlopeDifference( iCF ) = std::abs( U( iCF, iX, 1 ) - dU( iCF ) );

      // if slopes differ too much, replace
      if ( SlopeDifference( iCF ) >
           SlopeLimiter_Threshold * std::abs( U( iCF, iX, 0 ) ) )
      {
        for ( UInt k = 1; k < order; k++ )
        {
          U( iCF, iX, k ) = 0.0;
        }
        U( iCF, iX, 1 ) = dU( iCF );
        if ( order >= 3 ) U( iCF, iX, 2 ) = d2U( iCF );
      }
      /* Note we have limited this cell */
      LimitedCell( iX ) = 1;
    }
  }
}

/**
 * Limit the quadratic term.
 **/
void SlopeLimiter::LimitQuadratic( Kokkos::View<Real***> U,
                                   ModalBasis *Basis,
                                   Kokkos::View<Real[3]> d2w, UInt iX,
                                   UInt nNodes )
{

  Real Phi2 = 0.0;

  for ( UInt i = 0; i < 3; i++ )
  {
    Mult2( i ) = 0.0;
  }

  // --- Compute info for limiter ---
  for ( UInt iCF = 0; iCF < 3; iCF++ )
  {
    dU_c_L( iCF ) = U( iCF, iX - 1, 1 );
    dU_c_T( iCF ) = U( iCF, iX, 1 );
    dU_c_R( iCF ) = U( iCF, iX + 1, 1 );

    dU_v_L( iCF ) = Basis->BasisEval( U, iX, iCF, 0, true );
    dU_v_R( iCF ) = Basis->BasisEval( U, iX, iCF, nNodes + 1, true );

    // initialize characteristic forms
    dw_c_L( iCF ) = 0.0;
    dw_c_T( iCF ) = 0.0;
    dw_c_R( iCF ) = 0.0;

    dw_v_L( iCF ) = 0.0;
    dw_v_R( iCF ) = 0.0;
  }

  // --- Map limiter variables to characteristics ---

  // store w_.. = invR @ U_..
  MatMul( 1.0, R_inv, dU_c_L, 1.0, dw_c_L );
  MatMul( 1.0, R_inv, dU_c_T, 1.0, dw_c_T );
  MatMul( 1.0, R_inv, dU_c_R, 1.0, dw_c_R );

  MatMul( 1.0, R_inv, dU_v_L, 1.0, dw_v_L );
  MatMul( 1.0, R_inv, dU_v_R, 1.0, dw_v_R );

  // Limited Slopes
  for ( UInt iCF = 0; iCF < 3; iCF++ )
  {
    Phi2       = BarthJespersen( dw_v_L( iCF ), dw_v_R( iCF ), dw_c_L( iCF ),
                                 dw_c_T( iCF ), dw_c_R( iCF ), alpha );
    d2U( iCF ) = Phi2 * d2w( iCF ); // 2nd derivative
  }

  // Transform back to conserved quantities
  if ( CharacteristicLimiting_Option )
  {
    // d2U -> R d2U
    MatMul( 1.0, R, d2U, 1.0, Mult2 );

    for ( UInt iCF = 0; iCF < 3; iCF++ )
    {
      d2U( iCF ) = Mult2( iCF );
    }
  }

  for ( UInt iCF = 0; iCF < 3; iCF++ )
  {
    U( iCF, iX, 2 ) = d2U( iCF );
  }
}

/**
 * Return the cell average of a field iCF on cell iX.
 * The parameter `int extrapolate` designates how the cell average is computed.
 *  0  : Return stadnard cell average on iX
 *  -1 : Extrapolate polynomial from iX+1 into iX
 *  +1 : Extrapolate polynomial from iX-1 into iX
 **/
Real SlopeLimiter::CellAverage( Kokkos::View<Real***> U,
                                  GridStructure *Grid,
                                  ModalBasis *Basis, UInt iCF,
                                  UInt iX, int extrapolate )
{
  const UInt nNodes = Grid->Get_nNodes( );

  Real avg  = 0.0;
  Real mass = 0.0;
  Real X;

  // Used to set loop bounds
  int mult           = 1;
  UInt end   = nNodes;
  UInt start = 0;

  if ( extrapolate == -1 ) mult = 1;
  if ( extrapolate == +0 ) mult = 0;
  if ( extrapolate == +1 ) mult = 2;

  if ( extrapolate == 0 )
  {
    start = 0;
  }
  else
  {
    start = 2 + mult * nNodes;
  }
  end = start + nNodes - 1;

  for ( UInt iN = start; iN < end; iN++ )
  {
    X = Grid->NodeCoordinate( iX + extrapolate,
                             iN ); // Need the metric on target cell
    mass += Grid->Get_Weights( iN - start ) * Grid->Get_SqrtGm( X ) *
            Grid->Get_Widths( iX +
                             extrapolate ); // / Basis.BasisEval( U,
                                            // iX+extrapolate, 0, iN+1, false );
    avg += Grid->Get_Weights( iN - start ) *
           Basis->BasisEval( U, iX + extrapolate, iCF, iN + 1, false ) *
           Grid->Get_SqrtGm( X ) * Grid->Get_Widths( iX + extrapolate );
  }

  return avg / mass;
}

// LimitedCell accessor
int SlopeLimiter::Get_Limited( UInt iX ) const
{
  return LimitedCell( iX );
}
