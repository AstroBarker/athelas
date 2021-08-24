/**
 * Utility routines for fluid fields.
 * Riemann Solvers are here.
**/

#include <iostream>

/**
 * Return a component iCF of the flux vector.
 * TODO: Flux_Fluid needs streamlining
**/
double Flux_Fluid( double V, double P, unsigned int iCF )
{
  if ( iCF == 0 )
  {
    return - V;
  }
  else if ( iCF == 1 )
  {
    return + P;
  }
  else if ( iCF == 2 )
  {
    return P * V;
  }
  else{ // Error case. Shouldn't trigger.
    std::cout << "Please input a valid iCF! (0,1,2). " << std::endl;
    std::cerr << "Please input a valid iCF! (0,1,2). " << std::endl;
    exit(-1);
    return -1;
  }
}


/**
 * Gudonov style numerical flux. Constucts v* and p* states.
**/
void NumericalFlux_Gudonov( double vL, double vR, double pL, double pR, 
     double zL, double zR, double& Flux_U, double& Flux_P  )
{
  Flux_U = ( pL - pR + zR*vR + zL*vL ) / ( zR + zL );
  Flux_P = ( zR*pL + zL*pR + zL*zR * (vL - vR) ) / ( zR + zL );
}

// HLL

// ComputePrimitive

//Compute Auxilliary