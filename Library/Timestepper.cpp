/**
 * File     :  Timestepper.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : SSPRK timestepping routines
**/ 

#include <iostream>
#include <vector>

#include "omp.h"

#include "Error.h"
#include "Grid.h"
#include "DataStructures.h"
#include "SlopeLimiter.h"
#include "Fluid_Discretization.h"
#include "PolynomialBasis.h"
#include "Timestepper.h"

/**
 * The constructor creates the necessary data structures for time evolution.
**/
TimeStepper::TimeStepper( unsigned int nS, unsigned int tO, unsigned int pOrder,
    GridStructure& Grid, std::string BCond )
    : mSize( Grid.Get_nElements() + 2 * Grid.Get_Guard() ),
      nStages(nS),
      tOrder(tO),
      BC(BCond),
      a_jk(nStages, nStages),
      b_jk(nStages, nStages),
      SumVar_U( 3, mSize + 1, pOrder ),
      SumVar_X( mSize + 1, 0.0),
      U_s(nStages+1, DataStructure3D( 3, mSize, pOrder )),
      dU_s(nStages+1, DataStructure3D( 3, mSize, pOrder )),
      Grid_s(nStages+1, GridStructure( Grid.Get_nNodes(), 
        Grid.Get_nElements(), Grid.Get_Guard(), Grid.Get_xL(), Grid.Get_xR() )),
      Flux_q( 3, mSize + 1, Grid.Get_nNodes() ),
      dFlux_num( 3, mSize + 1 ),
      uCF_F_L( 3, mSize ),
      uCF_F_R( 3, mSize ),
      Flux_U( nStages + 1, std::vector<double>(mSize + 1,0.0) ),
      Flux_P( mSize + 1, 0.0),
      uCF_L(3, 0.0),
      uCF_R(3, 0.0)
{

  // --- Call Initialization ---
  InitializeTimestepper();

}


// Initialize arrays for timestepper
// TODO: Separate nStages from a tOrder
void TimeStepper::InitializeTimestepper( )
{

  if ( nStages == 1 )
  {
    a_jk(0,0) = 1.0;
    b_jk(0,0) = 1.0;
  }
  else if ( nStages == 2 )
  {
    a_jk(0,0) = 1.0;
    a_jk(1,0) = 0.5;
    a_jk(1,1) = 0.5;

    b_jk(0,0) = 1.0;
    b_jk(1,0) = 0.0;
    b_jk(1,1) = 0.5;
  }
  else if ( nStages == 3 )
  {
    a_jk(0,0) = 1.0;
    a_jk(1,0) = 0.75;
    a_jk(1,1) = 0.25;
    a_jk(2,0) = 1.0 / 3.0;
    a_jk(2,1) = 0.0;
    a_jk(2,2) = 2.0 / 3.0;

    b_jk(0,0) = 1.0;
    b_jk(1,0) = 0.0;
    b_jk(1,1) = 0.25;
    b_jk(2,0) = 0.0;
    b_jk(2,1) = 0.0;
    b_jk(2,2) = 2.0 / 3.0;
  }
  else
  {
    throw Error("\n === Please enter an appropriate number of SSPRK stages (1,2,3) === \n");
  }
  
}


/** 
 * Update Solution with SSPRK methods 
 * TODO: adjust for spherically symmetric
 * TODO: Remove dU (we only use dU_s)
**/
void TimeStepper::UpdateFluid( myFuncType ComputeIncrement, double dt, 
    DataStructure3D& U, GridStructure& Grid, ModalBasis& Basis,
    SlopeLimiter& S_Limiter )
{

  const unsigned int order = Basis.Get_Order();
  const unsigned int ilo   = Grid.Get_ilo();
  const unsigned int ihi   = Grid.Get_ihi();
 
  // double sum_x = 0.0;

  unsigned short int i;

  std::vector<std::vector<double>> StageData(nStages + 1, 
    std::vector<double>(ihi + 2, 0.0));

  SumVar_U.zero();

  U_s[0] = U;
  // StageData holds left interface positions
  for ( unsigned int iX = 0; iX <= ihi+1; iX++ )
  {
    StageData[0][iX] = Grid.Get_Centers(iX) - Grid.Get_Widths(iX) / 2.0;
  }
  
  for ( unsigned short int iS = 1; iS <= nStages; iS++ )
  {
    i = iS - 1;
    // re-zero the summation variable `SumVar`
    SumVar_U.zero();
    for ( unsigned int iX = 0; iX <= ihi+1; iX++ )
    {
      SumVar_X[iX] = 0.0;
    }

    // --- Inner update loop ---
    
    for ( unsigned int j = 0; j < iS; j++ )
    {
      ComputeIncrement( U_s[j], Grid_s[j], Basis, dU_s[j], Flux_q, dFlux_num, uCF_F_L, uCF_F_R, 
                        Flux_U[j], Flux_P, uCF_L, uCF_R, BC );

      
      // inner sum
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < order; k++ )
      {
        SumVar_U(iCF,iX,k) += a_jk(i,j) * U_s[j](iCF,iX,k) 
                        + dt * b_jk(i,j) * dU_s[j](iCF,iX,k);
      }

      for ( unsigned int iX = 0; iX <= ihi+1; iX++ )
      {
        SumVar_X[iX] += a_jk(i,j) * StageData[j][iX]
                     + dt * b_jk(i,j) * Flux_U[j][iX];
      }
      
    }
    U_s[iS] = SumVar_U;
    StageData[iS] = SumVar_X;
    Grid_s[iS].UpdateGrid( StageData[iS] );

    S_Limiter.ApplySlopeLimiter( U_s[iS], Grid_s[iS], Basis );
    
  }
  
  U = U_s[nStages-0];

  // Grid.UpdateGrid( StageData[nStages] );
  Grid = Grid_s[nStages];
  S_Limiter.ApplySlopeLimiter( U, Grid, Basis );

}