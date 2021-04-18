/**
 * Functions for polynomial basis
 **/

 #include <iostream>

#include "PolynomialBasis.h"
#include "LinearAlgebraModules.h"
#include "QuadratureLibrary.h"

// TODO: Some arrays to be replaced with vectors, maybe?

void SetNodes( unsigned int nNodes, double* nodes, double** node_mat )
{

  int k;
  for ( unsigned int j = 0; j < nNodes; j++ )
  {
    k = 0;
    for ( unsigned int i = 0; i < nNodes; i++ )
    {
      if ( i == j )
      {
        continue;
      }else{
        node_mat[k][j] = nodes[i];
        k++;
      }

    }
  }

}

double Lagrange( unsigned int nNodes, double x, double* nodes )
{
  double result = 1.0;
  for ( unsigned int i = 0; i < nNodes - 1; i++ )
  {
    result *= ( x - nodes[i] ) / ( nodes[nNodes - 1] - nodes[i] );
  }
  return result;
}


double dLagrange( unsigned int nNodes, double x, double* nodes )
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


int main( int argc, char* argv[] )
{
  //testing 

  unsigned int nNodes = 2;
  double* nodes = new double[nNodes];
  double* weights = new double[nNodes];

  LG_Quadrature( nNodes, nodes, weights );

  std::cout << Lagrange( nNodes, nodes[1], nodes );

  return 0;
}