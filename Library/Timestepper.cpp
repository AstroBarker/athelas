/**
 * File     :  Timestepper.cpp.cpp
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
#include "Timestepper.h"
#include "Fluid_Discretization.h"
#include "PolynomialBasis.h"

// Initialize arrays for timestepper
void InitializeTimestepper( const unsigned short int nStages, 
  DataStructure2D& a_jk, DataStructure2D& b_jk )
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


void UpdateFluid( myFuncType ComputeIncrement, double dt, 
  DataStructure3D& U, GridStructure& Grid, ModalBasis& Basis,
  DataStructure2D& a_jk, DataStructure2D& b_jk,
  std::vector<DataStructure3D>& U_s, std::vector<DataStructure3D>& dU_s, std::vector<GridStructure>& Grid_s,
  DataStructure3D& dU, DataStructure3D& SumVar, DataStructure3D& Flux_q, DataStructure2D& dFlux_num, 
  DataStructure2D& uCF_F_L, DataStructure2D& uCF_F_R, std::vector<std::vector<double>>& Flux_U, 
  std::vector<double>& Flux_P, std::vector<double> uCF_L, std::vector<double> uCF_R,
  const short unsigned int nStages, DataStructure3D& D, SlopeLimiter& S_Limiter,
  const std::string BC )
{

  const unsigned int order = Basis.Get_Order();
  const unsigned int ilo   = Grid.Get_ilo();
  const unsigned int ihi   = Grid.Get_ihi();
 
  // double sum_x = 0.0;

  unsigned short int i;

  std::vector<double> SumVar_X(ihi + 2, 0.0);
  std::vector<std::vector<double>> StageData(nStages + 1, 
    std::vector<double>(ihi + 2, 0.0));

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
    SumVar.zero();
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
        // TODO: Can I just replace SumVar += with U_s[iS](...) += ????
        SumVar(iCF,iX,k) += a_jk(i,j) * U_s[j](iCF,iX,k) 
                        + dt * b_jk(i,j) * dU_s[j](iCF,iX,k);
      }

      for ( unsigned int iX = 0; iX <= ihi+1; iX++ )
      {
        SumVar_X[iX] += a_jk(i,j) * StageData[j][iX]
                     + dt * b_jk(i,j) * Flux_U[j][iX];
      }
      
    }
    U_s[iS] = SumVar;
    StageData[iS] = SumVar_X;
    Grid_s[iS].UpdateGrid( StageData[iS] );

    S_Limiter.ApplySlopeLimiter( U_s[iS], Grid_s[iS], D );
    
  }
  
  U = U_s[nStages-0];

  Grid.UpdateGrid( StageData[nStages] );
  S_Limiter.ApplySlopeLimiter( U, Grid, D );

}