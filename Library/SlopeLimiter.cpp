/**
 * File     :  SlopeLimiter.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Classes for slope limters
 * Contains : SlopeLimiter
 * TODO: Properly switch to purely modal basis.
**/ 

#include <iostream>
#include "SlopeLimiter.h"
#include "SlopeLimiter_Utilities.h"
#include "CharacteristicDecomposition.h"
#include "LinearAlgebraModules.h"
#include "DataStructures.h"
#include "Error.h"

#include <cstdlib>     /* abs */


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
      TCI_Threshold(TCI_Threshold_val)
{
  // --- Initialize SLope Limiter structures ---

  dU  = new double[3];
  SlopeDifference = new double[3];

  // --- Initialize Characteristic matrices ---

  R     = new double[3*3];
  R_inv = new double[3*3];

}


// Apply slope limiter
void SlopeLimiter::ApplySlopeLimiter( DataStructure3D& U, GridStructure& Grid, 
  DataStructure3D& D )
{

  if ( order == 1 )
  {
    return;
  }

  
  double* a      = new double[3];
  double* b      = new double[3];
  double* c      = new double[3];
  double* tmp    = new double[3];
  double* Vals   = new double[3];
  
  const unsigned int nNodes = Grid.Get_nNodes();
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

  //DetectTroubledCells( Mesh, U, D )

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    // Check if TCI val is less than TCI_Threshold
    // unsigned int j = 0;
    // for ( unsigned int k = 0; k < order; k++ )
    // {
    //   if ( D(0,iX,k) < TCI_Threshold )
    //   {
    //     j++;
    //   }
    // }
    // if ( j == order ) continue;

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