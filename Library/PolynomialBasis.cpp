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
#include <algorithm> /* std::sort */
#include <math.h>       /* pow */

#include "DataStructures.h"
#include "Grid.h"
#include "LinearAlgebraModules.h"
#include "QuadratureLibrary.h"
#include "PolynomialBasis.h"
#include "Error.h"

// *** Taylor Methods ***

/** 
 * Return Taylor polynomial of given order
 *
 * Parameters:
 * -----------
 * eta : coordinate
 * eta_c: center of mass
 **/
double Taylor( unsigned int order, double eta, double eta_c )
{

  if ( order < 0 ) throw Error("Please enter a valid polynomial order.\n");

  // Handle constant and linear terms separately -- no need to exponentiate.
  if ( order == 0 )
  {
    return 1.0;
  }
  else if ( order == 1 )
  {
    return eta - eta_c;
  }
  else
  {
    return std::pow( eta - eta_c, order );
  }
}


/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < Psi_m, Phi_n >
 * <f,g> = \sum_q \rho_q f_Q g_q j^0 w_q
 * TODO: FINISH
**/
double InnerProduct( BasisFuncType f, DataStructure3D& Phi, 
  unsigned int m, unsigned int n, unsigned int iX, unsigned int nNodes, 
  double eta_c, DataStructure3D& uPF, GridStructure& Grid )
{
  double result = 0.0;
  double eta_q = 0.0;

  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    eta_q = Grid.Get_Nodes(iN); // TODO: Grid needs "ComputeMass" and "ComputeCenterOfMass"
    result += f( m, eta_q, eta_c ) * Phi( iX, iN+1, n )
           * Grid.Get_Weights(iN) * 1.0; // Density, volume
  }

  return result;

}


/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < Psi_m, Phi_n >
 * <f,g> = \sum_q \rho_q f_Q g_q j^0 w_q
 * TODO: FINISH
**/
double InnerProduct( DataStructure3D& Phi, 
  unsigned int n, unsigned int iX, unsigned int nNodes, double eta_c,
  DataStructure3D& uPF, GridStructure& Grid )
{
  double result = 0.0;

  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    result += Phi( iX, iN+1, n )  * Phi( iX, iN+1, n ) 
           * Grid.Get_Weights(iN) * uPF(0,iX,iN); // Density, volume
  }

  return result;

}


// Gram-Schmidt orthogonalization to Taylor basis
// TODO: FINISH
double OrthoTaylor( unsigned int order, unsigned int iX, double eta, double eta_c, DataStructure3D& Phi,
  DataStructure3D& uPF, GridStructure& Grid )
{

  double result      = 0.0;
  double phi_n       = 0.0;
  double numerator   = 0.0;

  double* denominator= new double[order]; // hold normalizations


  result = Taylor( order, eta, eta_c );

  if ( order == 0 )
  {
    result = 1.0;
  }
  else if ( order == 1 )
  {
    result = eta - eta_c;
  }
  else
  {
    for ( unsigned int i = 0; i < order; i++ )
    {
      numerator   = InnerProduct( Taylor, Phi, order-i, order, iX, order, eta_c, uPF, Grid); // TODO: make sure order-i is correct for GS
      denominator[i] = InnerProduct( Phi, i, iX, order, eta_c, uPF, Grid );
      phi_n = Taylor( i, eta, eta_c );
      result -= ( numerator / denominator[i] ) * phi_n;
    }
  }

  delete [] denominator;

  return result;

}


/**
 * Pre-compute the orthogonal Taylor basis terms. Phi(k,eta) will store
 * the expansion terms for each order k, stored at various points eta.
 * We store: (-0.5, {GL nodes}, 0.5) for a total of nNodes+2
**/
void InitializeTaylorBasis( DataStructure3D& Phi, DataStructure3D& uPF,
  GridStructure& Grid, unsigned int order, unsigned int nNodes )
{
  const unsigned int n_eta = order + 2;
  const unsigned int ilo   = Grid.Get_ilo();
  const unsigned int ihi   = Grid.Get_ihi();

  double eta_c = -1.0;

  // Compute eta_c

  double eta = 0.5;
  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  for ( unsigned int i_eta = 0; i_eta < n_eta; i_eta++ )
  for ( unsigned int k = 0; k < order; k++ )
  {

    if ( i_eta == 0 )
    {
      eta = -0.5;
    }
    else if ( i_eta == nNodes + 1 )
    {
      eta = +0.5;
    }
    else
    {
      eta = Grid.Get_Nodes(i_eta-1);
    }

    Phi(iX, i_eta, k) = OrthoTaylor( k, iX, eta, eta_c, Phi, uPF, Grid );
  }
}

// *** Lagrange Methods ***

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

    // Now, sort all but last value
    std::sort(nodes, nodes + nNodes-1);
  }

  // std::sort(nodes, nodes + nNodes-1);
}


// Lagrange interpolating polynomial
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


// Derivative of Lagrange polynomial
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

// *** Legendre Methods ***


// Legendre polynomials
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


// Derivative of Legendre polynomials
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


// Evaluate interpolating polynomial at a point
double Poly_Eval( unsigned int nNodes, double* nodes, double* data, double point )
{

  // TODO: Generalize this a bit in terms of a given basis, not just Lagrange
  long double s = 0.0;

  for ( unsigned int i = 0; i < nNodes; i++ )
  {
    s += data[i] * Lagrange( nNodes, point, i, nodes ); 
  }

  return s;
}