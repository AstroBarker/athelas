#ifndef FLUIDUTILITIES_H
#define FLUIDUTILITIES_H

double Flux_Fluid( double V, double P, unsigned int iCF );
void NumericalFlux_Gudonov( double vL, double vR, double pL, double pR, 
     double zL, double zR, double& Flux_U, double& Flux_P  );

#endif