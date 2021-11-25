/**
 * File     :  SlopeLimiter.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Classes for slope limters
 * Contains : SlopeLimiter
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
 * The constructor creates the matrices necessary for nodal/modal conversion
 * and structures for applying the slope limiter
**/
SlopeLimiter::SlopeLimiter( GridStructure& Grid, unsigned int numNodes, double SlopeLimiterThreshold, 
    unsigned int Beta_TVD_val, unsigned int Beta_TVB_val, 
    bool CharacteristicLimitingOption, bool TCIOption, 
    double TCI_Threshold_val )
    : nNodes(numNodes),
    SlopeLimiter_Threshold(SlopeLimiterThreshold),
    Beta_TVD(Beta_TVD_val),
    Beta_TVB(Beta_TVB_val),
    CharacteristicLimiting_Option(CharacteristicLimitingOption),
    TCI_Option(TCIOption),
    TCI_Threshold(TCI_Threshold_val), 
    U_M(3,3,numNodes)
{
  // --- Initialize SLope Limiter structures ---

  U_K = new double[3*(Grid.Get_nElements() + 2*Grid.Get_Guard())]; //(3,nX=2nG)
  // U_M = new double[3 * 3 * Grid.Get_nNodes()];
  dU  = new double[3];
  SlopeDifference = new double[3];

  // DataStructure3D U_M(3,3,nNodes);

  // --- Initialize Characteristic matrices ---

  R     = new double[3*3];
  R_inv = new double[3*3];

  // --- Initialize transformation matrices ---
  
  P     = new double[nNodes*nNodes];
  K     = new double[nNodes*nNodes];
  P_inv = new double[nNodes*nNodes];
  K_inv = new double[nNodes*nNodes];

  for ( unsigned int i = 0; i < nNodes; i++ )
  for ( unsigned int j = 0; j < nNodes; j++ )
  {
    P[i+nNodes*j]     = 0.0;
    K[i+nNodes*j]     = 0.0;
    P_inv[i+nNodes*j] = 0.0;
    K_inv[i+nNodes*j] = 0.0;
  }

  // --- Construct matrices ---

  // Mass matrix and inverse

  for ( unsigned int i = 0; i < nNodes; i++ )
  {
    P[i+nNodes*i] = 1.0 / ( 2.0*i + 1 );
    P_inv[i+nNodes*i] = ( 2.0*i + 1 );  
  }

  // Hold nodes
  double* Nodes     = new double[nNodes];
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    Nodes[iN]     = Grid.Get_Nodes( iN );
  }
  
  // Mapping matrix K = \int \ell(x) P(x)
  double x = 0.0;
  for ( unsigned int i = 0; i < nNodes; i++ )
  for ( unsigned int j = 0; j < nNodes; j++ )
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    x = Grid.Get_Nodes( iN );
    K[i+nNodes*j]     += Grid.Get_Weights( iN ) * Legendre( j, x )
                      * Lagrange( nNodes, x, i, Nodes ); 
    K_inv[i+nNodes*j] += Grid.Get_Weights( iN ) * Legendre( j, x ) 
                      * Lagrange( nNodes, x, i, Nodes );
  }

  // Invert K
  InvertMatrix( K_inv, nNodes );

  // -- Free Memory ---
  delete [] Nodes;

}


// Map from nodal to modal representation
void SlopeLimiter::NodalToModal( double* Un, double* result, int nNodes )
{

  double* tmp = new double[nNodes];
  for ( int i = 0; i < nNodes; i++ )
  {
    tmp[i]    = 0.0;
    result[i] = 0.0;
  }
  //P_inv @ (K @ Un)
  MatMul( nNodes, 1, nNodes, 1.0, K, 
    nNodes, Un, 1, 1.0, tmp, 1 );
  
  MatMul( nNodes, 1, nNodes, 1.0, P_inv, 
    nNodes, tmp, 1, 1.0, result, 1 ); // TODO: issues start here with `result` :::: better with Row major in cblas
  
  delete [] tmp;

}


// Map from nodal to modal representation
void SlopeLimiter::ModalToNodal( double* Um, double* result, int nNodes )
{

  double* tmp = new double[nNodes];
  for ( int i = 0; i < nNodes; i++ )
  {
    tmp[i]    = 0.0;
    result[i] = 0.0;
  }

  MatMul( nNodes, 1, nNodes, 1.0, P, 
    nNodes, Um, 1, 1.0, tmp, 1 );

  MatMul( nNodes, 1, nNodes, 1.0, K_inv, 
    nNodes, tmp, 1, 1.0, result, 1 );

  delete [] tmp;

}


// Apply slope limiter
void SlopeLimiter::ApplySlopeLimiter( DataStructure3D& U, GridStructure& Grid, 
  DataStructure3D& D )
{
  const unsigned int nNodes = Grid.Get_nNodes();

  if ( nNodes == 1 )
  {
    return;
  }

  
  double* a      = new double[3];
  double* b      = new double[3];
  double* c      = new double[3];
  double* tmp    = new double[3];
  double* Vals   = new double[3];
  double* NVals  = new double[nNodes];
  double* NVals2 = new double[nNodes];
  
  const unsigned int ilo = Grid.Get_ilo();
  const unsigned int ihi = Grid.Get_ihi();

  double sumvar  = 0.0;
  double sumvar2 = 0.0;

  for ( int i = 0; i < 3; i++ )
  {
    a[i]    = 0.0;
    b[i]    = 0.0;
    c[i]    = 0.0;
    tmp[i]  = 0.0;
    Vals[i] = 0.0;
  }

  for ( unsigned int i = 0; i < nNodes; i++ )
  {
    NVals[i]  = 0.0;
    NVals2[i] = 0.0;
  }

  // --- Apply troubled cell indicator ---
  // Exit if we don't need to limit slopes

  //DetectTroubledCells( Mesh, U, D )

  for ( unsigned int iX = ilo; iX < ihi; iX++ )
  {
    // Check if TCI val is less than TCI_Threshold
    // unsigned int j = 0;
    // for ( unsigned int iN = 0; iN < nNodes; iN++ )
    // {
    //   if ( D(0,iX,iN) < TCI_Threshold )
    //   {
    //     j++;
    //   }
    // }
    // if ( j == nNodes ) continue;

    for ( int i = 0; i < 3; i++ )
    {
      a[i]    = 0.0;
      b[i]    = 0.0;
      c[i]    = 0.0;
      tmp[i]  = 0.0;
      Vals[i] = 0.0;
    }

    for ( unsigned int i = 0; i < nNodes; i++ )
    {
      NVals[i]  = 0.0;
      NVals2[i] = 0.0;
    }

    // --- Setup U_M and U_K ---

    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      sumvar = 0.0;
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        sumvar += Grid.Get_Weights(iN) * U(iCF,iX,iN);
        NVals[iN] = U(iCF,iX,iN);
      }
      // Cell average of conserved fields
      U_K[iX + 3*iCF] = sumvar;
      

      // --- Nodal To Modal Respresentation ---
      for ( unsigned int i = 0; i < nNodes; i++ )
      {
        NVals2[i] = 0.0;
      }
      NodalToModal( NVals, NVals2, nNodes );
      
      sumvar = 0.0;
      sumvar2 = 0.0;

      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        // Hold modal representation
        U_M(iCF,0,iN) = NVals2[iN];

        // Compute Cell Average of Neighbors 
        sumvar  += Grid.Get_Weights(iN) * U(iCF, iX-1, iN);
        sumvar2 += Grid.Get_Weights(iN) * U(iCF, iX+1, iN);
      }
      
      // --- Cell Average of Neighbors ---

      U_M(iCF,1,0) = sumvar;
      U_M(iCF,2,0) = sumvar2;

    }

    // --- Characteristic Limiting Matrices ---
    // Note: using cell averages

    if ( CharacteristicLimiting_Option )
    {
      for ( int iCF = 0; iCF < 3; iCF++ )
      {
        Vals[iCF] = U_K[iX + 3*iCF];
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
      Vals[iCF] = U_M(iCF, 0, 1);
    }
    
    // store a = invR @ U_M(:,0,1)
    MatMul( 3, 1, 3, 1.0, R_inv, 
      3, Vals, 1, 1.0, a, 1 );

    // for b, and c, check boundary conditions
    // TODO: Ensure Slope limiter boundary conditions are good

    if ( iX == ilo )
    {
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        c[iCF] = 0.0;
        Vals[iCF] = Beta_TVD * U_M(iCF, 2, 0) - U_M(iCF, 0, 0);
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
        Vals[iCF] = Beta_TVD * U_M(iCF, 0, 0) - U_M(iCF, 1, 0);
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
        Vals[iCF] = Beta_TVD * U_M(iCF, 0, 0) - U_M(iCF, 1, 0);
      }

      MatMul( 3, 1, 3, 1.0, R_inv, 
        3, Vals, 1, 1.0, b, 1 );
      
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        c[iCF] = 0.0; // reset c storage
        Vals[iCF] = Beta_TVD * U_M(iCF, 2, 0) - U_M(iCF, 0, 0);
        // std::printf("%d %d %.5f\n", iCF, iX, Vals[iCF]);
      }

      MatMul( 3, 1, 3, 1.0, R_inv, 
        3, Vals, 1, 1.0, c, 1 );
    }

    // Limited SLopes
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      tmp[iCF] = 0.0;
      dU[iCF] = minmodB( a[iCF], b[iCF], c[iCF], Grid.Get_Widths(iX), Beta_TVB );
      // std::printf("%d %.5f %.5f %.5f %.5f\n", iCF, a[iCF], b[iCF], c[iCF], dU[iCF] );
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
      SlopeDifference[iCF] = std::abs( U_M(iCF,0,1) - dU[iCF] );
    

      // if slopes differ too much, replace
      
      if ( SlopeDifference[iCF] > SlopeLimiter_Threshold * std::abs( U_M(iCF,0,0) ) )
      {
        
        for (unsigned int iN = 1; iN < nNodes; iN++ )
        {
          U_M(iCF,0,iN) = 0.0;
        }
        U_M(iCF,0,1) = dU[iCF];        

        for ( unsigned int iN = 0; iN < nNodes; iN++ )
        {
          NVals[iN] = U_M(iCF,0,iN); //U_M(...) wrong?
          NVals2[iN] = 0.0;
        }
        
        ModalToNodal( NVals, NVals2, nNodes );
        for ( unsigned int iN = 0; iN < nNodes; iN++ )
        {
          U(iCF, iX, iN) = NVals2[iN];
        }

      }
      
      //TODO: Denoted LimitedCell[iCF, iX] = True

    }
    
  }

  delete [] Vals;
  delete [] NVals;
  delete [] NVals2;
  delete [] a;
  delete [] b;
  delete [] tmp;
  delete [] c;

}