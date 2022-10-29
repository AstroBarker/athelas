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

#include "Abstractions.hpp"
#include "Grid.h"

typedef Real BasisFuncType( unsigned int, Real, Real );

class ModalBasis
{
 public:
  ModalBasis( Kokkos::View<Real***> uCF, GridStructure *Grid,
              unsigned int pOrder, unsigned int nN, unsigned int nElements,
              unsigned int nGuard );
  Real Taylor( unsigned int order, Real eta, Real eta_c );
  Real dTaylor( unsigned int order, Real eta, Real eta_c );
  Real OrthoTaylor( const unsigned int order, const unsigned int iX,
                      const unsigned int i_eta, const Real eta, Real eta_c,
                      const Kokkos::View<Real***> uCF,
                      GridStructure *Grid, const bool derivative_option );
  Real InnerProduct( const unsigned int n, const unsigned int iX,
                       const Real eta_c, const Kokkos::View<Real***> uCF,
                       GridStructure *Grid );
  Real InnerProduct( const unsigned int m, const unsigned int n,
                       const unsigned int iX, const Real eta_c,
                       const Kokkos::View<Real***> uCF,
                       GridStructure *Grid );
  void InitializeTaylorBasis( const Kokkos::View<Real***> U,
                              GridStructure *Grid );
  void InitializeLegendreBasis( const Kokkos::View<Real***> uCF,
                                GridStructure *Grid );
  void CheckOrthogonality( const Kokkos::View<Real***> uCF,
                           GridStructure *Grid );
  Real BasisEval( Kokkos::View<Real***> U, const unsigned int iX,
                    const unsigned int iCF, const unsigned int i_eta,
                    const bool DerivativeOption ) const;
  void ComputeMassMatrix( const Kokkos::View<Real***> uCF,
                          GridStructure *Grid );

  Real Get_Phi( unsigned int iX, unsigned int i_eta, unsigned int k ) const;
  Real Get_dPhi( unsigned int iX, unsigned int i_eta, unsigned int k ) const;
  Real Get_MassMatrix( unsigned int iX, unsigned int k ) const;

  int Get_Order( ) const;

  Real Legendre( unsigned int order, Real x );
  Real dLegendre( unsigned int order, Real x );

 private:
  unsigned int nX;
  unsigned int order;
  unsigned int nNodes;
  unsigned int mSize;

  Kokkos::View<Real**> MassMatrix;

  Kokkos::View<Real***> Phi;
  Kokkos::View<Real***> dPhi;
};

#endif
