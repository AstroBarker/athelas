#pragma once
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

#include "abstractions.hpp"
#include "grid.hpp"

using BasisFuncType = double(int, double, double);

class ModalBasis {
 public:
  ModalBasis(poly_basis::poly_basis basis, View3D<double> uCF,
             GridStructure *grid, int pOrder, int nN, int nElements,
             bool density_weight);
  static auto taylor(int order, double eta, double eta_c) -> double;
  static auto d_taylor(int order, double eta, double eta_c) -> double;
  auto ortho(int order, int ix, int i_eta, double eta, double eta_c,
             const View3D<double> uCF, const GridStructure *grid,
             bool derivative_option) -> double;
  auto inner_product(int m, int n, int ix, double eta_c, View3D<double> uPF,
                     const GridStructure *grid) const -> double;
  auto inner_product(int n, int ix, double eta_c, View3D<double> uPF,
                     const GridStructure *grid) const -> double;
  void initialize_taylor_basis(View3D<double> U, GridStructure *grid);
  void initialize_basis(const View3D<double> uPF, const GridStructure *grid);
  void check_orthogonality(const View3D<double> uPF,
                           const GridStructure *grid) const;
  [[nodiscard]] auto basis_eval(View3D<double> U, int ix, int q,
                                int i_eta) const -> double;
  [[nodiscard]] auto basis_eval(View2D<double> U, int ix, int q,
                                int i_eta) const -> double;
  [[nodiscard]] auto basis_eval(View1D<double> U, int ix, int i_eta) const
      -> double;
  void compute_mass_matrix(const View3D<double> uPF, const GridStructure *grid);

  [[nodiscard]] auto get_phi(int ix, int i_eta, int k) const -> double;
  [[nodiscard]] auto phi() const noexcept -> View3D<double>;
  [[nodiscard]] auto get_d_phi(int ix, int i_eta, int k) const -> double;
  [[nodiscard]] auto dphi() const noexcept -> View3D<double>;
  [[nodiscard]] auto get_mass_matrix(int ix, int k) const -> double;

  [[nodiscard]] auto get_order() const noexcept -> int;

  // L2 projection from nodal to modal representation
  void project_nodal_to_modal(
      View3D<double> uCF, View3D<double> uPF, GridStructure *grid, int q,
      int ix, const std::function<double(double, int, int)> &nodal_func) const;

  // L2 projection from nodal to modal representation for all cells
  void project_nodal_to_modal_all_cells(
      View3D<double> uCF, View3D<double> uPF, GridStructure *grid, int q,
      const std::function<double(double, int, int)> &nodal_func) const;

  static auto legendre(int n, double x) -> double;
  static auto d_legendre(int order, double x) -> double;
  static auto legendre(int n, double x, double x_c) -> double;
  static auto d_legendre(int n, double x, double x_c) -> double;
  static auto d_legendre_n(int poly_order, int deriv_order, double x) -> double;

 private:
  int nX_;
  int order_;
  int nNodes_;
  int mSize_;
  bool density_weight_;

  View2D<double> mass_matrix_{};

  View3D<double> phi_{};
  View3D<double> dphi_{};

  double (*func_)(const int n, const double x, const double x_c);
  double (*dfunc_)(const int n, const double x, double const x_c);
};
