#ifndef POLYNOMIAL_BASIS_HPP_
#define POLYNOMIAL_BASIS_HPP_
/**
 * @file polynomial_basis.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Core polynomial basis functions
 *
 * @details Provides means to construct and evaluate bases
 *            - Legendre
 *            - Taylor
 */

#include <algorithm> // std::copy
#include <vector>

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "grid.hpp"

using BasisFuncType = Real( int, Real, Real );

class ModalBasis {
 public:
  ModalBasis( PolyBasis::PolyBasis basis, View3D<Real> uCF, GridStructure* Grid,
              int pOrder, int nN, int nElements, int nGuard );
  static auto Taylor( int order, Real eta, Real eta_c ) -> Real;
  static auto dTaylor( int order, Real eta, Real eta_c ) -> Real;
  auto Ortho( int order, int iX, int i_eta, Real eta, Real eta_c,
              View3D<Real> uCF, GridStructure* Grid, bool derivative_option )
      -> Real;
  auto InnerProduct( int m, int n, int iX, Real eta_c, View3D<Real> uPF,
                     GridStructure* Grid ) const -> Real;
  auto InnerProduct( int n, int iX, Real eta_c, View3D<Real> uPF,
                     GridStructure* Grid ) const -> Real;
  void InitializeTaylorBasis( View3D<Real> U, GridStructure* Grid );
  void InitializeBasis( View3D<Real> uCF, GridStructure* Grid );
  void CheckOrthogonality( View3D<Real> uCF, GridStructure* Grid ) const;
  [[nodiscard]] auto basis_eval( View3D<Real> U, int iX, int iCF,
                                 int i_eta ) const -> Real;
  [[nodiscard]] auto basis_eval( View2D<Real> U, int iX, int iCF,
                                 int i_eta ) const -> Real;
  [[nodiscard]] auto basis_eval( View1D<Real> U, int iX, int i_eta ) const
      -> Real;
  void ComputeMassMatrix( View3D<Real> uCF, GridStructure* Grid );

  [[nodiscard]] auto Get_Phi( int iX, int i_eta, int k ) const -> Real;
  [[nodiscard]] auto Get_dPhi( int iX, int i_eta, int k ) const -> Real;
  [[nodiscard]] auto Get_MassMatrix( int iX, int k ) const -> Real;

  [[nodiscard]] auto Get_Order( ) const noexcept -> int;

  static auto Legendre( int n, Real x ) -> Real;
  static auto dLegendre( int order, Real x ) -> Real;
  static auto Legendre( int n, Real x, Real x_c ) -> Real;
  static auto dLegendre( int n, Real x, Real x_c ) -> Real;
  static auto dLegendreN( int poly_order, int deriv_order, Real x ) -> Real;

 private:
  int nX;
  int order;
  int nNodes;
  int mSize;

  View2D<Real> MassMatrix{ };

  View3D<Real> Phi{ };
  View3D<Real> dPhi{ };

  Real ( *func )( const int n, const Real x, const Real x_c );
  Real ( *dfunc )( const int n, const Real x, Real const x_c );
};

#endif // POLYNOMIAL_BASIS_HPP_
