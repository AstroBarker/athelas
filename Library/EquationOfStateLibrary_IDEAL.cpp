/**
 * Ideal Equation of State Routines
**/

#include <math.h>       /* sqrt */

// Compute pressure assuming an ideal gas
double ComputePressureFromPrimitive_IDEAL( double Ev, double GAMMA=1.4 )
{
  return ( GAMMA - 1.0 ) * Ev;
}


double ComputePressureFromConserved_IDEAL( double Tau, double V, double Em_T, 
  double GAMMA=1.4 )
{
  double Em = Em_T - 0.5 * V*V;
  double Ev = Em / Tau;
  double P = (GAMMA - 1.0) * Ev;

  return P;
}


double ComputeSoundSpeedFromConserved_IDEAL( double Tau, double V, double Em_T, 
  double GAMMA=1.4 )
{
  double Em = Em_T - 0.5 * V*V;

  double Cs = sqrt( GAMMA * ( GAMMA - 1.0 ) * Em / Tau*Tau );
              //  / ( D + GAMMA * Ev ) )
  return Cs;
}