#ifndef QUADRATURELIBRARY_H
#define QUADRATURELIBRARY_H

double Jacobi_Matrix( int m, double* aj, double* bj );
void LG_Quadrature( int m, double* nodes, double* weights );

#endif