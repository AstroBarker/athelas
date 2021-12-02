/**
 * File     :  Fluid_Discretization.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : The main spatial discretization and update routines go here.
 *  Compute divergence term.
**/ 

#include <iostream>
#include <algorithm>

#include "omp.h"

#include "Error.h"
#include "PolynomialBasis.h"
#include "DataStructures.h"
#include "Grid.h"
#include "Fluid_Discretization.h"
#include "BoundaryConditionsLibrary.h"
#include "EquationOfStateLibrary_IDEAL.h"
#include "FluidUtilities.h"

// Compute the divergence of the flux term for the update
void ComputeIncrement_Fluid_Divergence( DataStructure3D& U, GridStructure& Grid, 
  DataStructure3D& dU, DataStructure3D& Flux_q, DataStructure2D& dFlux_num, 
  DataStructure2D& uCF_F_L, DataStructure2D& uCF_F_R, std::vector<double>& Flux_U, 
  std::vector<double>& Flux_P, std::vector<double> uCF_L, std::vector<double> uCF_R )
{
  // const unsigned int nX     = Grid.Get_nElements();
  const unsigned int nNodes = Grid.Get_nNodes();
  // const unsigned int nG     = Grid.Get_Guard();
  const unsigned int ilo    = Grid.Get_ilo();
  const unsigned int ihi    = Grid.Get_ihi();

  double P_L, P_R, Cs_L, Cs_R, lam_L, lam_R, P;

  double Poly_L, Poly_R;

  // const unsigned int nF_X   = nX + 1 + 2 * nG;  // Number of Interfaces
  // const unsigned int nCF_K  = 3 * nX           // Number of Fluid Fields in Domain
  // const unsigned int nCF_F  = 3 * nF_X        // Number of Fluid Fields on Interfaces

  // TODO: Is there a way to simplify the below?

  double* Nodes    = new double[nNodes];
  double* Nodes2   = new double[nNodes]; // copy...
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    Nodes[iN]   = Grid.Get_Nodes( iN );
    // We have a copy, Nodes2, which is permuted in the Volume term
    Nodes2[iN]  = Nodes[iN];
  }
  
  // These hold data on an element, for passing to the basis interpolation
  double* tmp_L = new double[nNodes]; // for interpolation
  double* tmp_R = new double[nNodes]; // for interpolation

  // --- Interpolate Conserved Variable to Interfaces ---
  
  // Left/Right face states
  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  for ( unsigned int iX = ilo; iX <= ihi+1; iX++ ) //TODO: Check that these bounds are correct
  {
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    { 
      tmp_L[iN] = U(iCF, iX-1, iN);
      tmp_R[iN] = U(iCF, iX-0, iN);
      // if( iCF == 2 ) std::cout << iN << " " << tmp_L[iN] << " " << U(iCF, iX-1, iN) << std::endl;
    }

    uCF_F_L(iCF, iX) = Poly_Eval( nNodes, Nodes, tmp_L, +0.5 );
    uCF_F_R(iCF, iX) = Poly_Eval( nNodes, Nodes, tmp_R, -0.5 );
    // if ( iCF == 0) std::printf("%d %.18f %.18f %.18f %.18f %.18f\n", iX, uCF_F_L(iCF, iX), uCF_F_L(iCF, iX), tmp_L[0], tmp_L[1], tmp_L[2] );
  }
  

  // --- Calc numerical flux at all faces
  // #pragma omp parallel for
  for ( unsigned int iX = ilo; iX <= ihi+1; iX++ ) //TODO: Bounds correct?
  {
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      uCF_L[iCF] = uCF_F_L(iCF, iX);
      uCF_R[iCF] = uCF_F_R(iCF, iX);
    }

    P_L   = ComputePressureFromConserved_IDEAL( uCF_L[0], uCF_L[1], uCF_L[2] );
    Cs_L  = ComputeSoundSpeedFromConserved_IDEAL( uCF_L[0], uCF_L[1], uCF_L[2] );
    lam_L = Cs_L / uCF_L[0];

    P_R   = ComputePressureFromConserved_IDEAL( uCF_R[0], uCF_R[1], uCF_R[2] );
    Cs_R  = ComputeSoundSpeedFromConserved_IDEAL( uCF_R[0], uCF_R[1], uCF_R[2] );
    lam_R = Cs_R / uCF_R[0];

    // --- Numerical Fluxes ---

    // Riemann Problem
    NumericalFlux_Gudonov( uCF_L[1], uCF_R[1], P_L, P_R, lam_L, lam_R, Flux_U[iX], Flux_P[iX] );
    // std::printf("%f %f %.18f %.18f %.18f\n", P_L, P_R, uCF_L[1], Flux_U[iX], uCF_R[1] );
    
    // TODO: Clean This Up
    dFlux_num(0, iX) = - Flux_U[iX];
    dFlux_num(1, iX) = + Flux_P[iX];
    dFlux_num(2, iX) = + Flux_U[iX] * Flux_P[iX];
    
  }
  
  // --- Surface Term ---

  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  // #pragma omp parallel for simd collapse(2)
  for ( unsigned int iX  = ilo-0; iX <= ihi+0; iX++ )
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    Poly_L = Lagrange( nNodes, - 0.5, iN, Nodes );
    Poly_R = Lagrange( nNodes, + 0.5, iN, Nodes );
    
    dU(iCF,iX,iN) += - ( + dFlux_num(iCF,iX+1) * Poly_R 
                         - dFlux_num(iCF,iX+0) * Poly_L );
  
    // Compute Flux_q everywhere for the Volume term

    P = ComputePressureFromConserved_IDEAL( U(0,iX,iN), U(1,iX,iN), U(2,iX,iN) );
    Flux_q(iCF,iX,iN) = Flux_Fluid( U(1,iX,iN), P, iCF );
  }
  
  // --- Volume Term ---

  double local_sum = 0.0;
  // double X1 = 0.0;
  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  // #pragma omp parallel for simd collapse(2)
  for ( unsigned int iX  = ilo-0; iX <= ihi+0; iX++ )
  for ( unsigned int iN  = 0; iN < nNodes; iN++ )
  {
    local_sum = 0.0;
    PermuteNodes(nNodes, iN, Nodes);
    for ( unsigned int i = 0; i < nNodes; i++ )
    {
      // X1 = Grid.NodeCoordinate(...)
      local_sum += Grid.Get_Weights(i) * Flux_q(iCF,iX,i) 
                * dLagrange( nNodes, Nodes2[i], Nodes );
    }
    std::sort(Nodes, Nodes + nNodes);

    dU(iCF,iX,iN) += local_sum;
  }

  // // TODO: Testing. 
  // // --- Calculate interface velocities using cell averaged velocities 
  // // Attempt use for Grid Update.
  // std::vector<double> Weights(nNodes);
  // for ( unsigned int iN = 0; iN < nNodes; iN++ )
  // {
  //   Weights[iN]   = Grid.Get_Weights( iN );
  // }
  // // --- Calc numerical flux at all faces
  // for ( unsigned int iX = ilo; iX <= ihi+1; iX++ ) //TODO: Bounds correct?
  // {
  //   for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  //   {
  //     uCF_L[iCF] = U.CellAverage( iCF, iX-1, nNodes, Weights );
  //     uCF_R[iCF] = U.CellAverage( iCF, iX-0, nNodes, Weights );
  //   }

  //   P_L   = ComputePressureFromConserved_IDEAL( uCF_L[0], uCF_L[1], uCF_L[2] );
  //   Cs_L  = ComputeSoundSpeedFromConserved_IDEAL( uCF_L[0], uCF_L[1], uCF_L[2] );
  //   lam_L = Cs_L / uCF_L[0];

  //   P_R   = ComputePressureFromConserved_IDEAL( uCF_R[0], uCF_R[1], uCF_R[2] );
  //   Cs_R  = ComputeSoundSpeedFromConserved_IDEAL( uCF_R[0], uCF_R[1], uCF_R[2] );
  //   lam_R = Cs_R / uCF_R[0];

  //   // --- Numerical Fluxes ---

  //   // Riemann Problem
  //   NumericalFlux_Gudonov( uCF_L[1], uCF_R[1], P_L, P_R, lam_L, lam_R, Flux_U[iX], Flux_P[iX] );
  // }


  // --- Deallocate ---
  delete [] Nodes;
  delete [] Nodes2;
  delete [] tmp_L;
  delete [] tmp_R;
}


/** Compute dU for timestep update. e.g., U = U + dU * dt
 * 
 * Parameters: 
 * -----------
 * U : Conserved variables
 * Grid : Grid object
 * dU : Update vector
 * Flux_q : Nodal fluxes, for volume term
 * dFLux_num : numerical surface flux
 * uCF_F_L, uCF_F_R : left/right face states
 * Flux_U, Flux_P : Fluxes (from Riemann problem)
 * BC : (string) boundary condition type
**/
void Compute_Increment_Explicit( DataStructure3D& U, GridStructure& Grid, 
  DataStructure3D& dU, DataStructure3D& Flux_q, DataStructure2D& dFlux_num, 
  DataStructure2D& uCF_F_L, DataStructure2D& uCF_F_R, std::vector<double>& Flux_U, 
  std::vector<double>& Flux_P, std::vector<double> uCF_L, std::vector<double> uCF_R,
  const std::string BC )
{

  const unsigned int nNodes = Grid.Get_nNodes();
  const unsigned int ilo    = Grid.Get_ilo();
  const unsigned int ihi    = Grid.Get_ihi();

  // --- Apply BC ---
  ApplyBC_Fluid( U, Grid, BC );

  // --- Detect Shocks ---
  //TODO: Code up a shock detector...

  // --- Compute Increment for new solution ---

  // --- First: Zero out dU. It is reused storage and we only increment it ---
  dU.zero();
  for ( unsigned int iX = 0; iX <= ihi+1; iX++ )
  {
    Flux_U[iX] = 0.0;
  }

  ComputeIncrement_Fluid_Divergence( U, Grid, dU, Flux_q, dFlux_num, 
    uCF_F_L, uCF_F_R, Flux_U, Flux_P, uCF_L, uCF_R );

  // ---

  // double X;
  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  // #pragma omp parallel for simd collapse(2)
  for ( unsigned int iX  = ilo-0; iX <= ihi+0; iX++ )
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    // X = Grid.NodeCoordinate(iX,iN);
    dU(iCF,iX,iN) /= ( Grid.Get_Weights(iN) * Grid.Get_Widths(iX) / U(0,iX,iN) );
  }

  // --- Increment Gravity --- ?
}