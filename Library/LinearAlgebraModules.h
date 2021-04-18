#ifndef LINEARALGEBRAMODULES_H
#define LINEARALGEBRAMODULES_H

double** AllocateMatrix( int n, int m );
void DeallocateMatrix( double** A, int rows );
double** matmul( double** A, double** B,
    int rows_A, int cols_A, int rows_B, int cols_B );
void Tri_Sym_Diag( int n, double* d, double* e, double* array );

#endif