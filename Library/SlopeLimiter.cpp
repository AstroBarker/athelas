/**
 * File     :  SlopeLimiter.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Classes for slope limters
 * Contains : SlopeLimiter
**/ 

#include <cstdlib>     /* abs */
#include <algorithm>   /* std::min, std::max */
#include <cstdlib>     /* abs */

#include <iostream>
#include "SlopeLimiter_Utilities.h"
#include "CharacteristicDecomposition.h"
#include "LinearAlgebraModules.h"
#include "DataStructures.h"
#include "Error.h"
#include "SlopeLimiter.h"


/**
 * The constructor creates the matrices structures for applying the slope limiter
**/
SlopeLimiter::SlopeLimiter( GridStructure& Grid, unsigned int pOrder, double SlopeLimiterThreshold, 
    unsigned int Beta_TVD_val, unsigned int Beta_TVB_val, 
    bool CharacteristicLimitingOption, bool TCIOption, 
    double TCI_Threshold_val )
    : order(pOrder),
      SlopeLimiter_Threshold(SlopeLimiterThreshold),
      Beta_TVD(Beta_TVD_val),
      Beta_TVB(Beta_TVB_val),
      CharacteristicLimiting_Option(CharacteristicLimitingOption),
      TCI_Option(TCIOption),
      TCI_Threshold(TCI_Threshold_val),
      D( 3, Grid.Get_nElements()+2*Grid.Get_Guard() )
{
  // --- Initialize SLope Limiter structures ---

  dU  = new double[3];
  SlopeDifference = new double[3];

  // --- Initialize Characteristic matrices ---

  R     = new double[3*3];
  R_inv = new double[3*3];

}


/**
 * Apply the Troubled Cell Indicator of Fu & Shu (2017) 
 * to flag cells for limiting
**/
void SlopeLimiter::DetectTroubledCells( DataStructure3D& U, 
  GridStructure& Grid, ModalBasis& Basis )
{
  const unsigned int ilo = Grid.Get_ilo();
  const unsigned int ihi = Grid.Get_ihi();

  double result      = 0.0;
  double cell_avg    = 0.0;
  double denominator = 0.0;

  // Cell averages by extrapolating L and R neighbors into current cell
  double cell_avg_L = 0.0;
  double cell_avg_R = 0.0;

  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    result = 0.0;
    cell_avg = U(iCF,iX,0);

    // Extrapolate neighboring poly representations into current cell
    // and compute the new cell averages
    cell_avg_L = CellAverage( U, Grid, Basis, iCF, iX, -1 );
    cell_avg_R = CellAverage( U, Grid, Basis, iCF, iX, +1 );
    // std::printf("%f %f %f \n", cell_avg_L, cell_avg, cell_avg_R);
    result += ( std::abs( cell_avg - cell_avg_L ) + std::abs( cell_avg - cell_avg_R ) );

    denominator = std::max( std::max( std::abs(cell_avg_L), 
      std::abs(cell_avg_R) ), cell_avg );
    
    D(iCF,iX) = result / denominator;
    // std::printf("%d %f %f\n", iCF, result, denominator);

  }
}


// Apply slope limiter
void SlopeLimiter::ApplySlopeLimiter( DataStructure3D& U, GridStructure& Grid, 
  ModalBasis& Basis )
{

  // Do not apply for first order method. No slopes!
  if ( order == 1 )
  {
    return;
  }

  
  double* a    = new double[3];
  double* b    = new double[3];
  double* c    = new double[3];
  double* tmp  = new double[3];
  double* Vals = new double[3];
  
  const unsigned int ilo    = Grid.Get_ilo();
  const unsigned int ihi    = Grid.Get_ihi();

  for ( int i = 0; i < 3; i++ )
  {
    a[i]    = 0.0;
    b[i]    = 0.0;
    c[i]    = 0.0;
    tmp[i]  = 0.0;
    Vals[i] = 0.0;
  }

  // --- Apply troubled cell indicator ---
  // Exit if we don't need to limit slopes

  if ( TCI_Option ) DetectTroubledCells( U, Grid, Basis );

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    // Check if TCI val is less than TCI_Threshold
    unsigned int j = 0;
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      // if ( iCF == 1 ) continue;
      if ( D(iCF,iX) > TCI_Threshold ) j++; // ! What is the appropriate data layout for D !
      // std::printf("%d %f\n", iCF, D(iCF,iX));
    }
    if ( j == 0 ) continue;

    for ( int i = 0; i < 3; i++ )
    {
      a[i]    = 0.0;
      b[i]    = 0.0;
      c[i]    = 0.0;
      tmp[i]  = 0.0;
      Vals[i] = 0.0;
    }


    // --- Characteristic Limiting Matrices ---
    // Note: using cell averages

    if ( CharacteristicLimiting_Option )
    {
      for ( int iCF = 0; iCF < 3; iCF++ )
      {
        Vals[iCF] = U(iCF,iX,0);
      }
      ComputeCharacteristicDecomposition( Vals, R, R_inv );
    }
    else
    {
      IdentityMatrix( R, 3 );
      IdentityMatrix( R_inv, 3 );
    }

    // multiply invR @ U_M[:,0,1] ( U_M[:,0,1] = slopes in modal basis)
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      a[iCF] = 0.0;
      Vals[iCF] = U(iCF, iX, 1);
    }
    
    // store a = invR @ U(:,iX,1)
    MatMul( 3, 1, 3, 1.0, R_inv, 
      3, Vals, 1, 1.0, a, 1 );

    // for b, and c, check boundary conditions
    // TODO: Ensure Slope limiter boundary conditions are good

    if ( iX == ilo )
    {
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        c[iCF] = 0.0;
        Vals[iCF] = Beta_TVD * U(iCF, iX+1, 0) - U(iCF, iX, 0);
      }

      MatMul( 3, 1, 3, 1.0, R_inv, 
        3, Vals, 1, 1.0, c, 1 );
      
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        b[iCF] = c[iCF];
      }
    }
    else if ( iX == ihi )
    {
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        b[iCF] = 0.0;
        Vals[iCF] = Beta_TVD * U(iCF, iX, 0) - U(iCF, iX-1, 0);
      }

      MatMul( 3, 1, 3, 1.0, R_inv, 
        3, Vals, 1, 1.0, b, 1 );
      
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        c[iCF] = b[iCF];
      }
    }
    else
    {
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        b[iCF] = 0.0;
        Vals[iCF] = Beta_TVD * U(iCF, iX, 0) - U(iCF, iX-1, 0);
      }

      MatMul( 3, 1, 3, 1.0, R_inv, 
        3, Vals, 1, 1.0, b, 1 );
      
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        c[iCF] = 0.0; // reset c storage
        Vals[iCF] = Beta_TVD * U(iCF, iX+1, 0) - U(iCF, iX, 0);
      }

      MatMul( 3, 1, 3, 1.0, R_inv, 
        3, Vals, 1, 1.0, c, 1 );
    }

    // Limited SLopes
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      tmp[iCF] = 0.0;
      dU[iCF] = minmodB( a[iCF], b[iCF], c[iCF], Grid.Get_Widths(iX), Beta_TVB );
    }

    // Transform back to conserved quantities
    if ( CharacteristicLimiting_Option )
    {
      // dU -> R dU
      MatMul( 3, 1, 3, 1.0, R, 
        3, dU, 1, 1.0, tmp, 1 );

      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        dU[iCF] = tmp[iCF];
      }
    }

    // --- Compare Limited to Original Slopes

    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      SlopeDifference[iCF] = std::abs( U(iCF, iX, 1) - dU[iCF] );
    

      // if slopes differ too much, replace
      
      if ( SlopeDifference[iCF] > SlopeLimiter_Threshold * std::abs( U(iCF, iX, 0) ) )
      {
        for (unsigned int k = 1; k < order; k++ )
        {
          U(iCF, iX, k) = 0.0;
        }
        U(iCF, iX, 1) = dU[iCF];
      }
      
      //TODO: Denoted LimitedCell[iCF, iX] = True

    }
    
  }

  delete [] Vals;
  delete [] a;
  delete [] b;
  delete [] tmp;
  delete [] c;

}


/**
 * Return the cell average of a field iCF on cell iX.
 * The parameter `int extrapolate` designates how the cell average is computed.
 *  0  : Return stadnard cell average on iX
 *  -1 : Extrapolate polynomial from iX-1 into iX
 *  +1 : Extrapolate polynomial from iX+1 into iX 
**/
double SlopeLimiter::CellAverage( DataStructure3D& U, GridStructure& Grid, ModalBasis& Basis,
  unsigned int iCF, unsigned int iX, int extrapolate )
{
  const unsigned int nNodes = Grid.Get_nNodes();

  double avg = 0.0;

  // Used to set loop bounds
  int mult  = 1;
  unsigned int end   = nNodes; 
  unsigned int start = 0;

  if ( extrapolate == -1 ) mult = 1;
  if ( extrapolate ==  0 ) mult = 0;
  if ( extrapolate == +1 ) mult = 2;

  start = 1 + mult * nNodes;
  end   = start + nNodes;

  for ( unsigned int iN = start; iN < end; iN++ )
  {
    avg += Grid.Get_Weights(iN-start) * Basis.BasisEval( U, iX, iCF, iN+1 );
  }

  return avg;
}