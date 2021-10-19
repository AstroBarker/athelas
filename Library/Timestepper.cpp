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
#include "Timestepper.h"
#include "Fluid_Discretization.h"
#include "PolynomialBasis.h"

// Initialize arrays for timestepper
void InitializeTimestepper( const unsigned short int nStages, 
  const int nX, const unsigned int nN,
  DataStructure2D& a_jk, DataStructure2D& b_jk,
  std::vector<DataStructure3D>& U_s, std::vector<DataStructure3D>& dU_s )
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
 * TODO: Figure out GridUpdate placement. Interstage or at end.
 * NOTE: `Flux_q` forms inner sum (3D) for timestep and Flux_q(3D) for increment
**/

void UpdateFluid( myFuncType ComputeIncrement, double dt, 
  DataStructure3D& U, GridStructure& Grid, 
  DataStructure2D& a_jk, DataStructure2D& b_jk,
  std::vector<DataStructure3D>& U_s, std::vector<DataStructure3D>& dU_s,
  DataStructure3D& dU, DataStructure3D& SumVar, DataStructure3D& Flux_q, DataStructure2D& dFlux_num, 
  DataStructure2D& uCF_F_L, DataStructure2D& uCF_F_R, std::vector<double>& Flux_U, 
  std::vector<double>& Flux_P, std::vector<double> uCF_L, std::vector<double> uCF_R,
  const short unsigned int nStages, const std::string BC )
{

  const unsigned int nNodes = Grid.Get_nNodes();
  const unsigned int ilo    = Grid.Get_ilo();
  const unsigned int ihi    = Grid.Get_ihi();

  const double frac = 1.0 / nStages;

  unsigned short int i;

  U_s[0] = U;
  
  for ( unsigned short int iS = 1; iS <= nStages; iS++ )
  {
    i = iS - 1;
    // re-zero the summation variable `SumVar`
    SumVar.zero();

    // --- Inner update loop ---
    
    for ( unsigned int j = 0; j < iS; j++ )
    {
      ComputeIncrement( U_s[j], Grid, dU_s[j], Flux_q, dFlux_num, uCF_F_L, uCF_F_R, 
                        Flux_U, Flux_P, uCF_L, uCF_R, BC );

      
      // inner sum
      for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        SumVar(iCF,iX,iN) += a_jk(i,j) * U_s[j](iCF,iX,iN) 
                        + dt * b_jk(i,j) * dU_s[j](iCF,iX,iN);
      }
      
    }
    U_s[iS] = SumVar;
    // Grid.UpdateGrid( U_s[iS], Flux_U, a_jk(nStages-1,i) * dt ); // ... works?
    Grid.UpdateGrid( U_s[iS], Flux_U, frac * dt );
    // TODO: ApplySlopeLimiter
  }
  
  U = U_s[nStages-0];

  // Grid.UpdateGrid( U, Flux_U, dt ); // Might switch to this

}