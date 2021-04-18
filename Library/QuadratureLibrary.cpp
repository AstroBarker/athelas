/* Functions for computing necessary quadrature rules */

#include <math.h>
#include <iostream>

#include "QuadratureLibrary.h"
#include "Constants.h"
#include "LinearAlgebraModules.h"

/**
 * Gauss-Legendre Quadrature
 **/


/**
 * Jacobi matrix for Legendre-Gauss quadrature rule
 *
 * Parameters:
 * 
 *   int m      : number of quadrature nodes
 *   double* aj : matrix diagonal    (output)
 *   double* bj : matrix subdiagonal (output)
 *   double z   : zero-th moment     (output)
 */
double Jacobi_Matrix( int m, double* aj, double* bj )
{

  double ab;
  double zemu;
  double abi;
  double abj;
  
  ab = 0.0;  
  zemu = 2.0 / ( ab + 1.0 );  
  for ( int i = 0; i < m; i++ )
  {
    aj[i] = 0.0;
  }

  for ( int i = 1; i <= m; i++ )
  {
    abi = i + ab * ( i % 2 );
    abj = 2 * i + ab;
    bj[i-1] = sqrt ( abi * abi / ( abj * abj - 1.0 ) );
  }

  return zemu;

}

void LG_Quadrature( int m, double* nodes, double* weights )
{
  double* aj = new double[m];
  double* bj = new double[m];

  double zemu;
  
  //  Get the Jacobi matrix and zero-th moment.
  zemu = Jacobi_Matrix( m, aj, bj );

  // Nodes and Weights
  for ( int i = 0; i < m; i++ )
  {
    nodes[i] = aj[i];
    // std::cout << aj[i] << " ";
  }
  
  weights[0] = sqrt ( zemu );
  for ( int i = 1; i < m; i++ )
  {
    weights[i] = 0.0;
  }
  
  //
  //  Diagonalize the Jacobi matrix.
  //
  Tri_Sym_Diag( m, nodes, bj, weights ); //imtqlx

  for ( int i = 0; i < m; i++ )
  {
    weights[i] = weights[i] * weights[i];
    // std::cout << weights[i] << " ";

    // Shift to interval [-0.5, 0.5]
    weights[i] *= 0.5;
    nodes[i]   *= 0.5;
  }

  delete [] aj;
  delete [] bj;

}

int main( int argc, char* argv[] )
{
  //testing
//   std::cout << "asfasf";
  int order = 11;

  double* weights = new double[order];
  double* nodes = new double[order];
  
  LG_Quadrature( order, nodes, weights );
  
  for ( int i = 0; i < order; i++)
  {
      std::cout << weights[i] << " ";
  }
  std::cout << std::endl;
  for ( int i = 0; i < order; i++)
  {
      std::cout << nodes[i] << " ";
  }

  return 0;
}