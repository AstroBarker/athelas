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
    GridStructure& Grid, bool Geometry, std::string BCond )
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
        Grid.Get_nElements(), Grid.Get_Guard(), Grid.Get_xL(), Grid.Get_xR(), 
        Geometry )),
      StageData(nStages + 1, 
        std::vector<double>(mSize + 1, 0.0)),
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

  if ( tOrder == 1 and nStages > 1 )
  {
    throw Error("\n === \n \
      Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n");
  }
  if ( (nStages != tOrder && nStages != 5) )
  {
    throw Error("\n === \n \
      Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n");
  }

  // Init to zero
  for ( unsigned int i = 0; i < nStages; i++ )
  for ( unsigned int j = 0; j < nStages; j++ )
  {
    a_jk(i,j) = 0.0;
    b_jk(i,j) = 0.0;
  }

  if ( nStages < 5  )
  {

    if ( tOrder == 1 )
    {
      a_jk(0,0) = 1.0;
      b_jk(0,0) = 1.0;
    }
    else if ( tOrder == 2 )
    {
      a_jk(0,0) = 1.0;
      a_jk(1,0) = 0.5;
      a_jk(1,1) = 0.5;

      b_jk(0,0) = 1.0;
      b_jk(1,0) = 0.0;
      b_jk(1,1) = 0.5;
    }
    else if ( tOrder == 3 )
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
  }
  else if ( nStages == 5 )
  {
    if ( tOrder == 1 )
    {
      throw Error("\n === We do support a 1st order, 5 stage SSPRK integrator. === \n");
    }
    else if ( tOrder == 2 )
    {
      a_jk(0,0) = 1.0;
      a_jk(4,0) = 0.2;
      a_jk(1,1) = 1.0;
      a_jk(2,2) = 1.0;
      a_jk(3,3) = 1.0;
      a_jk(4,4) = 0.8;

      b_jk(0,0) = 0.25;
      b_jk(1,1) = 0.25;
      b_jk(2,2) = 0.25;
      b_jk(3,3) = 0.25;
      b_jk(4,4) = 0.20;
    }
    else if ( tOrder == 3 )
    {
      a_jk(0,0) = 1.0;
      a_jk(1,0) = 0.0;
      a_jk(2,0) = 0.56656131914033;
      a_jk(3,0) = 0.09299483444413;
      a_jk(4,0) = 0.00736132260920;
      a_jk(1,1) = 1.0;
      a_jk(3,1) = 0.00002090369620;
      a_jk(4,1) = 0.20127980325145;
      a_jk(2,2) = 0.43343868085967;
      a_jk(4,2) = 0.00182955389682;
      a_jk(3,3) = 0.90698426185967;
      a_jk(4,4) = 0.78952932024253;

      b_jk(0,0) = 0.37726891511710;
      b_jk(3,0) = 0.00071997378654;
      b_jk(4,0) = 0.00277719819460;
      b_jk(1,1) = 0.37726891511710;
      b_jk(4,1) = 0.00001567934613;
      b_jk(2,2) = 0.16352294089771;
      b_jk(3,3) = 0.34217696850008;
      b_jk(4,4) = 0.29786487010104;
    }
    else if ( tOrder == 4 )
    {
      a_jk(0,0) = 1.0;
      a_jk(1,0) = 0.44437049406734;
      a_jk(2,0) = 0.62010185138540;
      a_jk(3,0) = 0.17807995410773;
      a_jk(4,0) = 0.00683325884039;
      a_jk(1,1) = 0.55562950593266;
      a_jk(2,2) = 0.37989814861460;
      a_jk(4,2) = 0.51723167208978;
      a_jk(3,3) = 0.82192004589227;
      a_jk(4,3) = 0.12759831133288;
      a_jk(4,4) = 0.34833675773694;

      b_jk(0,0) = 0.39175222700392;
      b_jk(1,1) = 0.36841059262959;
      b_jk(2,2) = 0.25189177424738;
      b_jk(3,3) = 0.54497475021237;
      b_jk(4,3) = 0.08460416338212;
      b_jk(4,4) = 0.22600748319395;
    }
  }
  
}


/** 
 * Update Solution with SSPRK methods 
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

  SumVar_U.zero();

  U_s[0] = U;
  Grid_s[0] = Grid;
  // StageData holds left interface positions
  for ( unsigned int iX = 0; iX <= ihi+1; iX++ )
  {
    StageData[0][iX] = Grid.Get_LeftInterface(iX);
  }
  
  for ( unsigned short int iS = 1; iS <= nStages; iS++ )
  {
    i = iS - 1;
    // re-zero the summation variables `SumVar`
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
      for ( unsigned int iX = 0; iX <= ihi+1; iX++ )
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
    Grid_s[iS].UpdateGrid( StageData[iS], U_s[iS] );

    S_Limiter.ApplySlopeLimiter( U_s[iS], Grid_s[iS], Basis );
    
  }
  
  U = U_s[nStages-0];

  Grid = Grid_s[nStages];
  S_Limiter.ApplySlopeLimiter( U, Grid, Basis );

}