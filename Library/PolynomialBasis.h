#ifndef POLYNOMIALBASIS_H
#define POLYNOMIALBASIS_H

/**
 * File     :  PolynomialBasis.h
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Functions for polynomial basis
 * Contains : Class for Taylor basis.
 * Also  Lagrange, Legendre polynomials, arbitrary degree.
 **/

#include <algorithm> // std::copy
#include <vector>

#include "DataStructures.h"

typedef double BasisFuncType( unsigned int, double, double );

class ModalBasis
{
 public:
  ModalBasis( DataStructure3D& uCF, GridStructure& Grid, unsigned int pOrder,
              unsigned int nN, unsigned int nElements, unsigned int nGuard );
  double Taylor( unsigned int order, double eta, double eta_c );
  double dTaylor( unsigned int order, double eta, double eta_c );
  double OrthoTaylor( unsigned int order, unsigned int iX, unsigned int i_eta,
                      double eta, double eta_c, DataStructure3D& uCF,
                      GridStructure& Grid, bool derivative_option );
  double InnerProduct( unsigned int n, unsigned int iX, double eta_c,
                       DataStructure3D& uCF, GridStructure& Grid );
  double InnerProduct( unsigned int m, unsigned int n, unsigned int iX,
                       double eta_c, DataStructure3D& uCF,
                       GridStructure& Grid );
  void InitializeTaylorBasis( DataStructure3D& U, GridStructure& Grid );
  void InitializeLegendreBasis( DataStructure3D& uCF, GridStructure& Grid );
  void CheckOrthogonality( DataStructure3D& uCF, GridStructure& Grid );
  double BasisEval( DataStructure3D& U, unsigned int iX, unsigned int iCF,
                    unsigned int i_eta, bool DerivativeOption );
  void ComputeMassMatrix( DataStructure3D& uCF, GridStructure& Grid );

  double Get_Phi( unsigned int iX, unsigned int i_eta, unsigned int k ) const;
  double Get_dPhi( unsigned int iX, unsigned int i_eta, unsigned int k ) const;
  double Get_MassMatrix( unsigned int iX, unsigned int k );

  int Get_Order( );

  double Legendre( unsigned int order, double x );
  double dLegendre( unsigned int order, double x );

 private:
  unsigned int nX;
  unsigned int order;
  unsigned int nNodes;
  unsigned int mSize;

  DataStructure2D MassMatrix;

  DataStructure3D Phi;
  DataStructure3D dPhi;
};

#endif
