/**
 * File     :  PolynomialBasis.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Functions for polynomial basis
 *  Contains Lagrange and Legendre polynomials, arbitrary degree.
 *
 * TODO: Poly_Eval needs generalizing. 
**/ 

#include <iostream>
#include <algorithm> // std::sort

#include "PolynomialBasis.h"
#include "LinearAlgebraModules.h"
#include "QuadratureLibrary.h"

/**
 * Permute node array. Used for passing to dLagrange.
 * Maybe not the best solution.
**/
void PermuteNodes( unsigned int nNodes, unsigned int iN, double* nodes )
{

  // First, swap last and iN-th values
  if ( iN < nNodes - 1)
  {
    double tmp1 = nodes[iN];
    double tmp2 = nodes[nNodes-1];

    nodes[nNodes-1] = tmp1;
    nodes[iN]       = tmp2;
    // std::cout << iN << ": " << tmp1 << " " << tmp2 << std::endl;
    // std::cout << iN << ": " << nodes[0] << " " << nodes[1] << " " << nodes[2] << std::endl;

    // Now, sort all but last value
    std::sort(nodes, nodes + nNodes-1);
  }

  // std::sort(nodes, nodes + nNodes-1);
}


double Lagrange
       ( unsigned int nNodes, double x, unsigned int p, double* nodes )
{
  double result = 1.0;
  for ( unsigned int i = 0; i < nNodes - 0; i++ )
  {
    if ( i == p ) continue;

    result *= ( x - nodes[i] ) / ( nodes[p] - nodes[i] );
    
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

  if ( nNodes == 0 )
  {
    return 1.0;
  }
  else if ( nNodes == 1 )
  {
    return 2.0 * x;
  }
  else
  {

    x *= 2.0; // This maps to intercal [-0.5, 0.5]

    double Pn, Pnm1; // P_n, P_{n-1}
    double Pnp1 = 0.0;

    Pnm1 = 1.0; // P_0
    Pn = x;    //  P_1
    for ( unsigned int i = 1; i < nNodes; i++ )
    {
      Pnp1 = 2.0 * x * Pn - Pnm1 - ( x * Pn - Pnm1) / (i+1);
      // Pnp1 = ( (2*i + 1) * x * Pn - i * Pnm1 ) / (i + 1);

      Pnm1 = Pn;
      Pn = Pnp1;
    }

    return Pn;
  }
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