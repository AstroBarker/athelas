#ifndef LINEARALGEBRAMODULES_H
#define LINEARALGEBRAMODULES_H

double** AllocateMatrix( int n, int m );
void DeallocateMatrix( double** A, int rows );
void IdentityMatrix( double* Mat, unsigned int n );
double** matmul( double** A, double** B,
    int rows_A, int cols_A, int rows_B, int cols_B );
void Tri_Sym_Diag( int n, double* d, double* e, double* array );
void InvertMatrix( double* M, unsigned int n );
void MatMul( int m, int n, int k, double alpha, double* A, 
  int lda, double* B, int ldb, double beta, double* C, int ldc );

#endif