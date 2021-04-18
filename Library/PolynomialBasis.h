#ifndef POLYNOMIALBASIS_H
#define POLYNOMIALBASIS_H

void SetNodes( unsigned int nNodes, double* nodes, double** node_mat );
double Lagrange( unsigned int nNodes, double x, double* nodes );
double dLagrange( unsigned int nNodes, double x, double* nodes );

#endif