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

#include "Grid.h"

typedef double BasisFuncType( unsigned int, double, double );

class ModalBasis
{
 public:
  ModalBasis( Kokkos::View<double***> uCF, GridStructure& Grid,
              unsigned int pOrder, unsigned int nN, unsigned int nElements,
              unsigned int nGuard );
  double Taylor( unsigned int order, double eta, double eta_c );
  double dTaylor( unsigned int order, double eta, double eta_c );
  double OrthoTaylor( const unsigned int order, const unsigned int iX,
                      const unsigned int i_eta, const double eta, double eta_c,
                      const Kokkos::View<double***> uCF,
                      const GridStructure& Grid, const bool derivative_option );
  double InnerProduct( const unsigned int n, const unsigned int iX,
                       const double eta_c, const Kokkos::View<double***> uCF,
                       const GridStructure& Grid );
  double InnerProduct( const unsigned int m, const unsigned int n,
                       const unsigned int iX, const double eta_c,
                       const Kokkos::View<double***> uCF,
                       const GridStructure& Grid );
  void InitializeTaylorBasis( const Kokkos::View<double***> U,
                              const GridStructure& Grid );
  void InitializeLegendreBasis( const Kokkos::View<double***> uCF,
                                const GridStructure& Grid );
  void CheckOrthogonality( const Kokkos::View<double***> uCF,
                           const GridStructure& Grid );
  double BasisEval( Kokkos::View<double***> U, const unsigned int iX,
                    const unsigned int iCF, const unsigned int i_eta,
                    const bool DerivativeOption ) const;
  void ComputeMassMatrix( const Kokkos::View<double***> uCF,
                          const GridStructure& Grid );

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
