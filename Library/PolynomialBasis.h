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

#include "Kokkos_Core.hpp"

typedef double BasisFuncType( unsigned int, double, double );

class ModalBasis
{
 public:
  ModalBasis( Kokkos::View<double***> uCF, GridStructure& Grid,
              unsigned int pOrder, unsigned int nN, unsigned int nElements,
              unsigned int nGuard );
  double Taylor( unsigned int order, double eta, double eta_c );
  double dTaylor( unsigned int order, double eta, double eta_c );
  double OrthoTaylor( unsigned int order, unsigned int iX, unsigned int i_eta,
                      double eta, double eta_c, Kokkos::View<double***> uCF,
                      GridStructure& Grid, bool derivative_option );
  double InnerProduct( unsigned int n, unsigned int iX, double eta_c,
                       Kokkos::View<double***> uCF, GridStructure& Grid );
  double InnerProduct( unsigned int m, unsigned int n, unsigned int iX,
                       double eta_c, Kokkos::View<double***> uCF,
                       GridStructure& Grid );
  void InitializeTaylorBasis( Kokkos::View<double***> U, GridStructure& Grid );
  void InitializeLegendreBasis( Kokkos::View<double***> uCF,
                                GridStructure& Grid );
  void CheckOrthogonality( Kokkos::View<double***> uCF, GridStructure& Grid );
  double BasisEval( Kokkos::View<double***> U, unsigned int iX,
                    unsigned int iCF, unsigned int i_eta,
                    bool DerivativeOption ) const;
  void ComputeMassMatrix( Kokkos::View<double***> uCF, GridStructure& Grid );

  double Get_Phi( unsigned int iX, unsigned int i_eta, unsigned int k ) const;
  double Get_dPhi( unsigned int iX, unsigned int i_eta, unsigned int k ) const;
  double Get_MassMatrix( unsigned int iX, unsigned int k ) const;

  int Get_Order( ) const;

  double Legendre( unsigned int order, double x );
  double dLegendre( unsigned int order, double x );

 private:
  unsigned int nX;
  unsigned int order;
  unsigned int nNodes;
  unsigned int mSize;

  Kokkos::View<double**> MassMatrix;

  Kokkos::View<double***> Phi;
  Kokkos::View<double***> dPhi;
};

#endif
