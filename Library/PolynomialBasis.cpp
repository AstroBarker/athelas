/**
 * Functions for polynomial basis
 **/

#include <iostream>

#include "PolynomialBasis.h"
#include "LinearAlgebraModules.h"
#include "QuadratureLibrary.h"

double Lagrange
       ( unsigned int nNodes, double x, unsigned int p, double* nodes )
{
  double result = 1.0;
  for ( unsigned int i = 0; i < nNodes - 0; i++ )
  {
    if ( i == p )
    {
      continue;
    }else{
    result *= ( x - nodes[i] ) / ( nodes[p] - nodes[i] );
    }
  }
  return result;
}


double dLagrange
       ( unsigned int nNodes, double x, double* nodes )
{
  double denominator = 1.0;
  for ( unsigned int i = 0; i < nNodes - 1; i++ )
  {
    denominator *= ( nodes[nNodes-1] - nodes[i] );
  }

  double numerator;
  double result = 0.0;
  for ( unsigned int i = 0; i < nNodes - 1; i++ )
  {
    numerator = 1.0;
    for ( unsigned int j = 0; j < nNodes - 1; j++ )
    {
      if ( j == i ) continue;

      numerator *= ( x - nodes[j] );
    }
    result += numerator / denominator;
  }
  return result;
}


double Legendre( unsigned int nNodes, double x )
{
  x *= 2.0; // This maps to intercal [-0.5, 0.5]

  double Pn, Pnm1; // P_n, P_{n-1}
  double Pnp1 = 0.0;

  Pnm1 = 1.0; // P_0
  Pn = x;    //  P_1
  for ( unsigned int i = 0; i < nNodes; i++ )
  {
    Pnp1 = 2.0 * x * Pn - Pnm1 - ( x * Pn - Pnm1) / (i+1);

    Pnm1 = Pn;
    Pn = Pnp1;
  }

  return Pn;
}


double dLegendre( unsigned int nNodes, double x )
{

  double dPn; // P_n
  // double dPnp1 = 0.0;

  dPn = 0.0;
  for ( unsigned int i = 0; i < nNodes; i++ )
  {
    dPn = ( i + 1 ) * Legendre( i, x ) + 2.0 * x * dPn;
  }

  return dPn;
}


double Poly_Eval( unsigned int nNodes, double* nodes, double* data, double point )
{

  // TODO: Generalize this a bit in terms of a given basis, not just Lagrange
  double s = 0.0;

  for ( unsigned int i = 0; i < nNodes; i++ )
  {
    s += data[i] * Lagrange( nNodes, point, i, nodes ); 
  }

  return s;
}