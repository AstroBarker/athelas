/**
 * File     :  EquationOfStateLibrary_IDEAL.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Ideal equation of state routines
 **/

#include <math.h> /* sqrt */

#include "PolynomialBasis.h"
#include "EquationOfStateLibrary_IDEAL.h"

// Compute pressure assuming an ideal gas
double ComputePressureFromPrimitive_IDEAL( const double Ev, const double GAMMA )
{
  return ( GAMMA - 1.0 ) * Ev;
}

double ComputePressureFromConserved_IDEAL( const double Tau, const double V,
                                           const double Em_T,
                                           const double GAMMA )
{
  double Em = Em_T - 0.5 * V * V;
  double Ev = Em / Tau;
  double P  = ( GAMMA - 1.0 ) * Ev;

  return P;
}

double ComputeSoundSpeedFromConserved_IDEAL( const double Tau, const double V,
                                             const double Em_T,
                                             const double GAMMA )
{
  double Em = Em_T - 0.5 * V * V;

  double Cs = sqrt( GAMMA * ( GAMMA - 1.0 ) * Em );
  //  / ( D + GAMMA * Ev ) )
  return Cs;
}

// nodal specific internal energy
double ComputeInternalEnergy( const Kokkos::View<double***> U,
                              const ModalBasis& Basis, const unsigned int iX,
                              const unsigned int iN )
{
  double Vel = Basis.BasisEval( U, iX, 1, iN, false );
  double EmT = Basis.BasisEval( U, iX, 2, iN, false );

  return EmT - 0.5 * Vel * Vel;
}

// cell average specific internal energy
double ComputeInternalEnergy( const Kokkos::View<double***> U,
                              const unsigned int iX )
{
  return U( 2, iX, 0 ) - 0.5 * U( 1, iX, 0 ) * U( 1, iX, 0 );
}