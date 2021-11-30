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
#include "DataStructures.h"
#include "Grid.h"
#include "PolynomialBasis.h"
#include "Fluid_Discretization.h"
#include "BoundaryConditionsLibrary.h"
#include "EquationOfStateLibrary_IDEAL.h"
#include "FluidUtilities.h"

// Compute the divergence of the flux term for the update
void ComputeIncrement_Fluid_Divergence( DataStructure3D& U, GridStructure& Grid, 
  ModalBasis& Basis, DataStructure3D& dU, DataStructure3D& Flux_q, 
  DataStructure2D& dFlux_num, DataStructure2D& uCF_F_L, 
  DataStructure2D& uCF_F_R, std::vector<double>& Flux_U, 
  std::vector<double>& Flux_P, std::vector<double> uCF_L, std::vector<double> uCF_R )
{
  const unsigned int nNodes = Grid.Get_nNodes();
  const unsigned int ilo    = Grid.Get_ilo();
  const unsigned int ihi    = Grid.Get_ihi();

  double P_L, P_R, Cs_L, Cs_R, lam_L, lam_R, P;

  double Poly_L, Poly_R;

  // const unsigned int nF_X   = nX + 1 + 2 * nG;  // Number of Interfaces
  // const unsigned int nCF_K  = 3 * nX           // Number of Fluid Fields in Domain
  // const unsigned int nCF_F  = 3 * nF_X        // Number of Fluid Fields on Interfaces

  // --- Interpolate Conserved Variable to Interfaces ---
  
  // Left/Right face states
  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  for ( unsigned int iX  = ilo; iX <= ihi+1; iX++ ) //TODO: Check that these bounds are correct
  {
    uCF_F_L(iCF, iX) = Basis.BasisEval( U, iX-1, iCF, nNodes + 1 );
    uCF_F_R(iCF, iX) = Basis.BasisEval( U, iX, iCF, 0 );
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
    
    // TODO: Clean This Up
    dFlux_num(0, iX) = - Flux_U[iX];
    dFlux_num(1, iX) = + Flux_P[iX];
    dFlux_num(2, iX) = + Flux_U[iX] * Flux_P[iX];
    
  }
  
  // --- Surface Term ---

  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  // #pragma omp parallel for simd collapse(2)
  for ( unsigned int iX = ilo-0; iX <= ihi+0; iX++ )
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    
    Poly_L = Basis.Get_Phi(iX, 0, iN);
    Poly_R = Basis.Get_Phi(iX, nNodes+1, iN);
    
    dU(iCF,iX,iN) += - ( + dFlux_num(iCF,iX+1) * Poly_R 
                         - dFlux_num(iCF,iX+0) * Poly_L );
    
    // Compute Flux_q everywhere for the Volume term

    // TODO: Lookie here. What is Flux_q (_,_,:)? 
    // TODO: HERE: This integral needs GL quadrature - need to evaluate U at GL nodes and pass to Flux_q
    P = ComputePressureFromConserved_IDEAL( Basis.BasisEval( U, iX, 0, iN+1 ), 
      Basis.BasisEval( U, iX, 1, iN+1 ), Basis.BasisEval( U, iX, 2, iN+1 ) );
    Flux_q(iCF,iX,iN) = Flux_Fluid( Basis.BasisEval( U, iX, 1, iN+1 ), P, iCF ); // Basis.BasisEval( U, iX, iCF, 0 )
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
    // PermuteNodes(nNodes, iN, Nodes);
    for ( unsigned int i = 0; i < nNodes; i++ )
    {
      // X1 = Grid.NodeCoordinate(...)
      local_sum += Grid.Get_Weights(i) * Flux_q(iCF,iX,i) 
                * Basis.Get_dPhi( iX, i+1, iN );
                // * dLagrange( nNodes, Nodes2[i], Nodes );
    }
    // std::sort(Nodes, Nodes + nNodes);

    dU(iCF,iX,iN) += local_sum;
  }

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
  ModalBasis& Basis, DataStructure3D& dU, DataStructure3D& Flux_q, 
  DataStructure2D& dFlux_num, DataStructure2D& uCF_F_L, 
  DataStructure2D& uCF_F_R, std::vector<double>& Flux_U, 
  std::vector<double>& Flux_P, std::vector<double> uCF_L, 
  std::vector<double> uCF_R, const std::string BC )
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

  ComputeIncrement_Fluid_Divergence( U, Grid, Basis, dU, Flux_q, dFlux_num, 
    uCF_F_L, uCF_F_R, Flux_U, Flux_P, uCF_L, uCF_R );

  // ---

  // double X;
  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  for ( unsigned int iX  = ilo-0; iX <= ihi+0; iX++ )
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    // X = Grid.NodeCoordinate(iX,iN);
    dU(iCF,iX,iN) /= Basis.Get_MassMatrix( iX, iN );
  }

  // --- Increment Gravity --- ?
}