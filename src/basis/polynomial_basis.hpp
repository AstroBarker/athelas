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
 *            - legendre
 *            - taylor
 */

#include <algorithm> // std::copy
#include <vector>

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "grid.hpp"

using BasisFuncType = Real( int, Real, Real );

class ModalBasis {
 public:
  ModalBasis( poly_basis::poly_basis basis, View3D<Real> uCF,
              GridStructure* Grid, int pOrder, int nN, int nElements,
              int nGuard );
  static auto taylor( int order, Real eta, Real eta_c ) -> Real;
  static auto d_taylor( int order, Real eta, Real eta_c ) -> Real;
  auto ortho( int order, int iX, int i_eta, Real eta, Real eta_c,
              View3D<Real> uCF, GridStructure* Grid, bool derivative_option )
      -> Real;
  auto inner_product( int m, int n, int iX, Real eta_c, View3D<Real> uPF,
                      GridStructure* Grid ) const -> Real;
  auto inner_product( int n, int iX, Real eta_c, View3D<Real> uPF,
                      GridStructure* Grid ) const -> Real;
  void initialize_taylor_basis( View3D<Real> U, GridStructure* Grid );
  void initialize_basis( View3D<Real> uCF, GridStructure* Grid );
  void check_orthogonality( View3D<Real> uCF, GridStructure* Grid ) const;
  [[nodiscard]] auto basis_eval( View3D<Real> U, int iX, int iCF,
                                 int i_eta ) const -> Real;
  [[nodiscard]] auto basis_eval( View2D<Real> U, int iX, int iCF,
                                 int i_eta ) const -> Real;
  [[nodiscard]] auto basis_eval( View1D<Real> U, int iX, int i_eta ) const
      -> Real;
  void compute_mass_matrix( View3D<Real> uCF, GridStructure* Grid );

  [[nodiscard]] auto get_phi( int iX, int i_eta, int k ) const -> Real;
  [[nodiscard]] auto get_d_phi( int iX, int i_eta, int k ) const -> Real;
  [[nodiscard]] auto get_mass_matrix( int iX, int k ) const -> Real;

  [[nodiscard]] auto get_order( ) const noexcept -> int;

  static auto legendre( int n, Real x ) -> Real;
  static auto d_legendre( int order, Real x ) -> Real;
  static auto legendre( int n, Real x, Real x_c ) -> Real;
  static auto d_legendre( int n, Real x, Real x_c ) -> Real;
  static auto d_legendre_n( int poly_order, int deriv_order, Real x ) -> Real;

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
