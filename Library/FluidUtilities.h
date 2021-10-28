#ifndef FLUIDUTILITIES_H
#define FLUIDUTILITIES_H

double Flux_Fluid( double V, double P, unsigned int iCF );
double Fluid( double Tau, double V, double Em_T, int iCF );
void NumericalFlux_Gudonov( double vL, double vR, double pL, double pR, 
     double zL, double zR, double& Flux_U, double& Flux_P  );
void NumericalFlux_HLL( double tauL, double tauR, double vL, double vR, 
  double eL, double eR, double pL, double pR, double zL, double zR, 
  int iCF, double& out );
double ComputeTimestep_Fluid( DataStructure3D& U, 
     GridStructure& Grid, const double CFL );

#endif