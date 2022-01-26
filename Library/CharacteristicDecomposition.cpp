/**
 * File     :  CharacteristicDecomposition.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Compute matrices for characteristic decomposition
**/ 

#include <iostream>
#include <math.h>       /* sqrt */

#include "CharacteristicDecomposition.h"
#include "Error.h"
// #include "EquationOfStateLibrary_IDEAL.h"

void ComputeCharacteristicDecomposition( double* U, double* R, double* R_inv )
{

  const double Tau  = U[0];
  const double V    = U[1];
  const double Em_T = U[2];

  // const double P  = ComputePressureFromConserved_IDEAL( Tau, V, Em_T );
  // const double Cs = ComputeSoundSpeedFromConserved_IDEAL( Tau, V, Em_T );

  const double Em = Em_T - 0.5 * V*V;

  const float GAMMA = 1.4;

  const double k      = std::sqrt( GAMMA * ( GAMMA - 1.0 ) );
  const double sqrt_e = std::sqrt( Em );
  const double InvTau = 1.0 / Tau;

  /* --- Thermodynamic Derivatives of Pressure --- */

  // const double P_Tau = - (GAMMA - 1.0) * Em / ( Tau * Tau );
  // const double P_Em  = + (GAMMA - 1.0) / Tau;
  // const double P_T_E = - Em / Tau; // ratio of derivatives

  // Eigenvalues are rho * Cs...
  // const double lam = std::sqrt( P * P_Em - P_Tau );

  /*  --- Compute Matrix Elements --- */

  for ( int i = 0; i < 3; i++ ) R[i] = 1.0;
  
  R[3] = + sqrt_e * k * InvTau;
  R[4] = + 0.0;
  R[5] = - sqrt_e * k * InvTau;

  R[6] = (Em + sqrt_e * k * V - Em * GAMMA) * InvTau;
  R[7] = (Em * InvTau);
  R[8] = (Em - sqrt_e * k * V - Em * GAMMA) * InvTau;

  R_inv[0] = 0.5;
  R_inv[1] = Tau * (k * V + sqrt_e * GAMMA) / (2.0 * Em * k);
  R_inv[2] = - Tau / (2.0 * Em);

  R_inv[3] = GAMMA - 1.0;
  R_inv[4] = - V * Tau / Em;
  R_inv[5] = Tau / Em;

  R_inv[6] = 0.5;
  R_inv[7] = Tau * (k * V - sqrt_e * GAMMA) / (2.0 * Em * k);
  R_inv[8] = - Tau / (2.0 * Em);

  for ( int i = 0; i < 9; i++ )
  {
    R_inv[i] /= GAMMA;
  }
}