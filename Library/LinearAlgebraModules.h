#ifndef LINEARALGEBRAMODULES_H
#define LINEARALGEBRAMODULES_H

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"

void IdentityMatrix( Kokkos::View<Real**> Mat, unsigned int n );
Real** matmul( Real** A, Real** B, int rows_A, int cols_A, int rows_B,
                 int cols_B );
void Tri_Sym_Diag( int n, Real* d, Real* e, Real* array );
void InvertMatrix( Real* M, unsigned int n );
void MatMul( Real alpha, Kokkos::View<Real[3][3]> A,
             Kokkos::View<Real[3]> x, Real beta,
             Kokkos::View<Real[3]> y );

#endif
