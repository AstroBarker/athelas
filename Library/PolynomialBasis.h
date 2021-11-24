#ifndef POLYNOMIALBASIS_H
#define POLYNOMIALBASIS_H

typedef double BasisFuncType ( unsigned int, double, double );

double Taylor( unsigned int order, double eta, double eta_c );
double OrthoTaylor( unsigned int order, unsigned int iX, 
  double eta, double eta_c, DataStructure3D& uPF, GridStructure& Grid );
double InnerProduct( DataStructure3D& Phi, 
  unsigned int n, unsigned int iX, unsigned int nNodes, double eta_c,
  DataStructure3D& uPF, GridStructure& Grid );
double InnerProduct( BasisFuncType f, DataStructure3D& Phi, 
  unsigned int m, unsigned int n, unsigned int iX, unsigned int nNodes, 
  double eta_c, DataStructure3D& uPF, GridStructure& Grid );
void InitializeTaylorBasis( DataStructure3D& Phi, DataStructure3D& U, 
  GridStructure& Grid, unsigned int order, unsigned int nNodes );
void CheckOrthogonality( DataStructure3D& Phi, DataStructure3D& uPF,
  GridStructure& Grid, unsigned int order, unsigned int nNodes );
double BasisEval( DataStructure3D& U, DataStructure3D& Phi, 
  unsigned int iX, unsigned int iCF, unsigned int i_eta, unsigned int order );
void PermuteNodes( unsigned int nNodes, unsigned int iN, double* nodes );
double Lagrange ( unsigned int nNodes, double x, unsigned int p, double* nodes );
double dLagrange( unsigned int nNodes, double x, double* nodes );
double Legendre ( unsigned int nNodes, double x );
double dLegendre( unsigned int nNodes, double x );

double Poly_Eval( unsigned int nNodes, double* nodes, double* data, double point );

#endif