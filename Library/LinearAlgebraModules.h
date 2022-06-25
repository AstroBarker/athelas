#ifndef LINEARALGEBRAMODULES_H
#define LINEARALGEBRAMODULES_H

#include "Kokkos_Core.hpp"

void IdentityMatrix( Kokkos::View<double**> Mat, unsigned int n );
double** matmul( double** A, double** B, int rows_A, int cols_A, int rows_B,
                 int cols_B );
void Tri_Sym_Diag( int n, double* d, double* e, double* array );
void InvertMatrix( double* M, unsigned int n );
void MatMul( double alpha, Kokkos::View<double[3][3]> A,
             Kokkos::View<double[3]> x, double beta,
             Kokkos::View<double[3]> y );

#endif