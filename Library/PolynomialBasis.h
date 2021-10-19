#ifndef POLYNOMIALBASIS_H
#define POLYNOMIALBASIS_H

void PermuteNodes( unsigned int nNodes, unsigned int iN, double* nodes );
double Lagrange ( unsigned int nNodes, double x, unsigned int p, double* nodes );
double dLagrange( unsigned int nNodes, double x, double* nodes );
double Legendre ( unsigned int nNodes, double x );
double dLegendre( unsigned int nNodes, double x );

double Poly_Eval( unsigned int nNodes, double* nodes, double* data, double point );

#endif