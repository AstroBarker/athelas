#ifndef POLYNOMIAL_BASIS_HPP_
#define POLYNOMIAL_BASIS_HPP_

/**
 * File     :  polynomial_basis.hpp
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

#include "abstractions.hpp"
#include "grid.hpp"

typedef Real BasisFuncType( int, Real, Real );

class ModalBasis {
 public:
  ModalBasis( PolyBasis::PolyBasis basis, const View3D<Real> uCF,
              GridStructure *Grid, const int pOrder, const int nN,
              const int nElements, const int nGuard );
  static Real Taylor( const int order, const Real eta, const Real eta_c );
  static Real dTaylor( const int order, const Real eta, const Real eta_c );
  Real Ortho( const int order, const int iX, const int i_eta, const Real eta,
              const Real eta_c, const View3D<Real> uCF, GridStructure *Grid,
              const bool derivative_option );
  Real InnerProduct( const int m, const int n, const int iX, const Real eta_c,
                     const View3D<Real> uPF, GridStructure *Grid );
  Real InnerProduct( const int n, const int iX, const Real eta_c,
                     const View3D<Real> uPF, GridStructure *Grid );
  void InitializeTaylorBasis( const View3D<Real> U, GridStructure *Grid );
  void InitializeBasis( const PolyBasis::PolyBasis basis,
                        const View3D<Real> uCF, GridStructure *Grid );
  void CheckOrthogonality( const View3D<Real> uCF, GridStructure *Grid );
  Real basis_eval( View3D<Real> U, const int iX, const int iCF,
                   const int i_eta ) const;
  Real basis_eval( View2D<Real> U, const int iX, const int iCF,
                   const int i_eta ) const;
  Real basis_eval( View1D<Real> U, const int iX, const int i_eta ) const;
  void ComputeMassMatrix( const View3D<Real> uCF, GridStructure *Grid );

  Real Get_Phi( const int iX, const int i_eta, const int k ) const;
  Real Get_dPhi( const int iX, const int i_eta, const int k ) const;
  Real Get_MassMatrix( const int iX, const int k ) const;

  int Get_Order( ) const;

  static Real Legendre( const int order, const Real x );
  static Real dLegendre( const int order, const Real x );
  static Real Legendre( const int order, const Real x, const Real x_c );
  static Real dLegendre( const int order, const Real x, const Real x_c );
  static Real dLegendreN( const int poly_order, const int deriv_order,
                          const Real x );

 private:
  int nX;
  int order;
  int nNodes;
  int mSize;

  View2D<Real> MassMatrix;

  View3D<Real> Phi;
  View3D<Real> dPhi;

  Real ( *func )( const int n, const Real x, const Real x_c );
  Real ( *dfunc )( const int n, const Real x, Real const x_c );
};

#endif // POLYNOMIAL_BASIS_HPP_
