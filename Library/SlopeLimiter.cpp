/**
 * File     :  SlopeLimiter.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Classes for slope limters
 * Contains : SlopeLimiter
 * ! Warning: ApplySlopeLimiter is a mess. !
 * TODO: Clean up ApplySlopeLimiter
**/ 

// BUG: !!
// I need to limit the quadratic term FIRST. 
// compute Phi2. Change U(:,:,2)
// Then, with the updated U, we can compute phi 1
// This will be better! Will it fix my issues?
// Maybe not. But I am doing it wrong.
// BUG: !!

#include <cstdlib>     /* abs */
#include <algorithm>   /* std::min, std::max */
#include <cstdlib>     /* abs */

#include <iostream>
#include "SlopeLimiter_Utilities.h"
#include "CharacteristicDecomposition.h"
#include "LinearAlgebraModules.h"
#include "Grid.h"
#include "DataStructures.h"
#include "Error.h"
#include "PolynomialBasis.h"
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

  alpha = 0.9;

  dU  = new double[3];
  d2U = new double[3];
  SlopeDifference = new double[3];

  U_c_L = new double[3];
  U_c_T = new double[3];
  U_c_R = new double[3];
  U_v_L = new double[3];
  U_v_R = new double[3];

  dU_c_L = new double[3];
  dU_c_T = new double[3];
  dU_c_R = new double[3];
  dU_v_L = new double[3];
  dU_v_R = new double[3];

  // characteristic forms
  w_c_L = new double[3];
  w_c_T = new double[3];
  w_c_R = new double[3];
  w_v_L = new double[3];
  w_v_R = new double[3];

  dw_c_L = new double[3];
  dw_c_T = new double[3];
  dw_c_R = new double[3];
  dw_v_L = new double[3];
  dw_v_R = new double[3];

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
    cell_avg_L = CellAverage( U, Grid, Basis, iCF, iX+1, -1 ); // from right
    cell_avg_R = CellAverage( U, Grid, Basis, iCF, iX-1, +1 ); // from left

    // Hacking
    if ( iCF == 0 )
    {
      cell_avg   = 1.0 / cell_avg;
      cell_avg_L = 1.0 / cell_avg_L;
      cell_avg_R = 1.0 / cell_avg_R;
    }
    else if ( iCF == 2 )
    {
      cell_avg   /= U(0,iX,0);
      cell_avg_L /= CellAverage( U, Grid, Basis, iCF, iX+1, -1 ); 
      cell_avg_R /= CellAverage( U, Grid, Basis, iCF, iX-1, +1 ); 
    }
    else
    {
      cell_avg *= 1.0;
    }

    result += ( std::abs( cell_avg - cell_avg_L ) 
           + std::abs( cell_avg - cell_avg_R ) );

    denominator = std::max( std::max( std::abs(cell_avg_L), 
      std::abs(cell_avg_R) ), cell_avg );
    
    D(iCF,iX) = result / denominator;

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

  // TODO: These should be in the class, so as to avoid reallocation
  double Phi1  = 0.0;
  double Phi2  = 0.0;
  
  double* a     = new double[3]; // holds char. slopes
  double* b     = new double[3];
  double* c     = new double[3];
  double* d     = new double[3]; // holds char. averages
  double* e     = new double[3]; // holds char. 2nd derivs
  double* tmp   = new double[3];
  double* Vals  = new double[3];
  double* Vals2 = new double[3];

  const unsigned int ilo = Grid.Get_ilo();
  const unsigned int ihi = Grid.Get_ihi();
  const unsigned int nNodes = Grid.Get_nNodes();

  for ( int i = 0; i < 3; i++ )
  {
    a[i]     = 0.0;
    b[i]     = 0.0;
    c[i]     = 0.0;
    d[i]     = 0.0;
    e[i]     = 0.0;
    tmp[i]   = 0.0;
    Vals[i]  = 0.0;
    Vals2[i] = 0.0;
  }

  // --- Apply troubled cell indicator ---
  // Exit if we don't need to limit slopes

  if ( TCI_Option ) DetectTroubledCells( U, Grid, Basis );

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    // Check if TCI val is less than TCI_Threshold
    int j = 0;
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      if ( D(iCF,iX) > TCI_Threshold && TCI_Option ) j++; // ! What is the appropriate data layout for D !
      // std::printf("%f %f\n", D(iCF,iX), TCI_Threshold);
    }
    
    if ( j == 0 && TCI_Option ) continue;

    for ( int i = 0; i < 3; i++ )
    {
      a[i]     = 0.0;
      b[i]     = 0.0;
      c[i]     = 0.0;
      d[i]     = 0.0;
      e[i]     = 0.0;
      tmp[i]   = 0.0;
      Vals[i]  = 0.0;
      Vals2[i] = 0.0;
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
      d[iCF] = 0.0;
      Vals[iCF]  = U(iCF, iX, 1); // Slopes
      Vals2[iCF] = U(iCF, iX, 0); // Cell averages
      // if ( order >= 3 ) tmp[iCF] = U(iCF, iX, 2); // 2nd derivative
    }
    
    // store a = invR @ dU
    MatMul( 3, 1, 3, 1.0, R_inv, 
      3, Vals, 1, 1.0, a, 1 );

    // store d = invR @ avg(U)
    MatMul( 3, 1, 3, 1.0, R_inv, 
      3, Vals2, 1, 1.0, d, 1 );
    

    // for b, and c, check boundary conditions
    // TODO: Ensure Slope limiter boundary conditions are good
    // ! May need to remove this once we fully switch to new limiter !
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

    // --- Limit Quadratic Term ---
    if ( order >= 3 )
    {
      LimitQuadratic( U, Basis, iX, nNodes);
      // store e = invR @ d2U
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        tmp[iCF] = U(iCF, iX, 2); // 2nd derivative
      }
      MatMul( 3, 1, 3, 1.0, R_inv, 
        3, tmp, 1, 1.0, e, 1 );
    }

    // --- Compute info for limiter ---
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      Vals[iCF]  = 0.0;
      Vals2[iCF] = 0.0;
      tmp[iCF]   = 0.0;

      U_c_L[iCF] = U(iCF, iX-1, 0);
      U_c_T[iCF] = U(iCF, iX  , 0);
      U_c_R[iCF] = U(iCF, iX+1, 0);

      U_v_L[iCF] = Basis.BasisEval( U, iX, iCF, 0, false );
      U_v_R[iCF] = Basis.BasisEval( U, iX, iCF, nNodes + 1, false );

      // initialize characteristic forms
      w_c_L[iCF] = 0.0;
      w_c_T[iCF] = 0.0;
      w_c_R[iCF] = 0.0;

      w_v_L[iCF] = 0.0;
      w_v_R[iCF] = 0.0;

    }

    // --- Map limiter variables to characteristics ---

    // store w_.. = invR @ U_..
    MatMul( 3, 1, 3, 1.0, R_inv, 
      3, U_c_L, 1, 1.0, w_c_L, 1 );
    MatMul( 3, 1, 3, 1.0, R_inv, 
      3, U_c_T, 1, 1.0, w_c_T, 1 );
    MatMul( 3, 1, 3, 1.0, R_inv, 
      3, U_c_R, 1, 1.0, w_c_R, 1 );

    MatMul( 3, 1, 3, 1.0, R_inv, 
      3, U_v_L, 1, 1.0, w_v_L, 1 );
    MatMul( 3, 1, 3, 1.0, R_inv, 
      3, U_v_R, 1, 1.0, w_v_R, 1 );


    // TODO: Apply Phi's to places
    // Limited Slopes
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      Phi1 = BarthJespersen( w_v_L[iCF], w_v_R[iCF], 
        w_c_L[iCF], w_c_T[iCF], w_c_R[iCF], alpha );
      
      dU[iCF] = Phi1 * a[iCF]; // Multiply slope by Phi1
      if ( order >= 3 ) d2U[iCF] = Phi1 * e[iCF]; // 2nd derivative
      
    }

    // Transform back to conserved quantities
    if ( CharacteristicLimiting_Option )
    {
      // dU -> R dU
      MatMul( 3, 1, 3, 1.0, R, 
        3, dU, 1, 1.0, tmp, 1 );
      // d2U -> R d2U
      MatMul( 3, 1, 3, 1.0, R, 
        3, d2U, 1, 1.0, Vals, 1 );

      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        dU[iCF]   = tmp[iCF];
        d2U[iCF]  = Vals[iCF];
      }
    }

    // --- Compare Limited to Original Slopes

    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      SlopeDifference[iCF] = std::abs( U(iCF, iX, 1) - dU[iCF] );
    

     // std::printf("%.9f %.9f\n", U(iCF,iX,2), d2U[iCF]);
      // if slopes differ too much, replace
      
      if ( SlopeDifference[iCF] > SlopeLimiter_Threshold * std::abs( U(iCF, iX, 0) ) )
      {
        for ( unsigned int k = 1; k < order; k++ )
        {
          U(iCF, iX, k) = 0.0;
        }
        U(iCF, iX, 1) = dU[iCF];
        if ( order >= 3 ) U(iCF,iX,2) = d2U[iCF];
      }
      
      //TODO: Denoted LimitedCell[iCF, iX] = True

    }
    
  }

  delete [] Vals;
  delete [] Vals2;
  delete [] a;
  delete [] b;
  delete [] tmp;
  delete [] c;
  delete [] d;
  delete [] e;

}


/**
 * Limit the quadratic term.
**/
void SlopeLimiter::LimitQuadratic( DataStructure3D& U, ModalBasis& Basis, 
  unsigned int iX, unsigned int nNodes )
{

  double Phi2 = 0.0;

  double* tmp = new double[3];
  for ( unsigned int i = 0; i < 3; i++ )
  {
    tmp[i] = 0.0;
  }

  // --- Compute info for limiter ---
  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  {
    dU_c_L[iCF] = U(iCF, iX-1, 1);
    dU_c_T[iCF] = U(iCF, iX  , 1);
    dU_c_R[iCF] = U(iCF, iX+1, 1);

    dU_v_L[iCF] = Basis.BasisEval( U, iX, iCF, 0, true );
    dU_v_R[iCF] = Basis.BasisEval( U, iX, iCF, nNodes + 1, true );   

    // initialize characteristic forms
    dw_c_L[iCF] = 0.0;
    dw_c_T[iCF] = 0.0;
    dw_c_R[iCF] = 0.0;

    dw_v_L[iCF] = 0.0;
    dw_v_R[iCF] = 0.0;
  }

  // --- Map limiter variables to characteristics ---

  // store w_.. = invR @ U_..
  MatMul( 3, 1, 3, 1.0, R_inv, 
    3, dU_c_L, 1, 1.0, dw_c_L, 1 );
  MatMul( 3, 1, 3, 1.0, R_inv, 
    3, dU_c_T, 1, 1.0, dw_c_T, 1 );
  MatMul( 3, 1, 3, 1.0, R_inv, 
    3, dU_c_R, 1, 1.0, dw_c_R, 1 );

  MatMul( 3, 1, 3, 1.0, R_inv, 
    3, dU_v_L, 1, 1.0, dw_v_L, 1 );
  MatMul( 3, 1, 3, 1.0, R_inv, 
    3, dU_v_R, 1, 1.0, dw_v_R, 1 );
  
  // TODO: Apply Phi's to places
  // Limited Slopes
  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  {
    Phi2 = BarthJespersen( dw_v_L[iCF], dw_v_R[iCF], 
      dw_c_L[iCF], dw_c_T[iCF], dw_c_R[iCF], alpha );
    d2U[iCF] = Phi2 * dw_c_T[iCF]; // 2nd derivative
    
  }

  // Transform back to conserved quantities
  if ( CharacteristicLimiting_Option )
  {
    // d2U -> R d2U
    MatMul( 3, 1, 3, 1.0, R, 
      3, d2U, 1, 1.0, tmp, 1 );

    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      d2U[iCF] = tmp[iCF];
    }
  }

  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  {
    U(iCF,iX,2) = d2U[iCF];
  }

  // --- deallocate ---
  delete [] tmp;
}



/**
 * Return the cell average of a field iCF on cell iX.
 * The parameter `int extrapolate` designates how the cell average is computed.
 *  0  : Return stadnard cell average on iX
 *  -1 : Extrapolate polynomial from iX+1 into iX
 *  +1 : Extrapolate polynomial from iX-1 into iX 
**/
double SlopeLimiter::CellAverage( DataStructure3D& U, GridStructure& Grid, ModalBasis& Basis,
  unsigned int iCF, unsigned int iX, int extrapolate )
{
  const unsigned int nNodes = Grid.Get_nNodes();

  double avg = 0.0;
  double vol = 0.0;
  double X;

  // Used to set loop bounds
  int mult  = 1;
  unsigned int end   = nNodes; 
  unsigned int start = 0;

  if ( extrapolate == -1 ) mult = 1;
  if ( extrapolate ==  0 ) mult = 0;
  if ( extrapolate == +1 ) mult = 2;

  start = 2 + mult * nNodes;
  end   = start + nNodes - 1;

  for ( unsigned int iN = start; iN < end; iN++ )
  {
    X = Grid.NodeCoordinate(iX+extrapolate,iN); // Need the metric on target cell
    vol += Grid.Get_Weights(iN-start) * Grid.Get_SqrtGm(X) 
        * Grid.Get_Widths(iX+extrapolate);// / Basis.BasisEval( U, iX, 0, iN+1, false );
    avg += Grid.Get_Weights(iN-start) * Basis.BasisEval( U, iX, iCF, iN+1, false ) 
        * Grid.Get_SqrtGm(X) * Grid.Get_Widths(iX+extrapolate);// / Basis.BasisEval( U, iX, 0, iN+1, false );
  }

  return avg / vol;
}
