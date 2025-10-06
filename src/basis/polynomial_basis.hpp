#pragma once

#include "basic_types.hpp"
#include "geometry/grid.hpp"
#include "kokkos_types.hpp"

namespace athelas::basis {

using BasisFuncType = double(int, double, double);

class ModalBasis {
 public:
  ModalBasis(poly_basis basis, AthelasArray3D<double> uCF, GridStructure *grid,
             int pOrder, int nN, int nElements, bool density_weight);
  static auto taylor(int order, double eta, double eta_c) -> double;
  static auto d_taylor(int order, double eta, double eta_c) -> double;
  auto ortho(int order, int ix, int i_eta, double eta, double eta_c,
             const AthelasArray3D<double> uCF, const GridStructure *grid,
             bool derivative_option) -> double;
  auto inner_product(int m, int n, int ix, double eta_c,
                     AthelasArray3D<double> uPF,
                     const GridStructure *grid) const -> double;
  auto inner_product(int n, int ix, double eta_c, AthelasArray3D<double> uPF,
                     const GridStructure *grid) const -> double;
  void initialize_taylor_basis(AthelasArray3D<double> U, GridStructure *grid);
  void initialize_basis(const AthelasArray3D<double> uPF,
                        const GridStructure *grid);
  void check_orthogonality(const AthelasArray3D<double> uPF,
                           const GridStructure *grid) const;
  [[nodiscard]] auto basis_eval(AthelasArray3D<double> U, int ix, int q,
                                int i_eta) const -> double;
  [[nodiscard]] auto basis_eval(AthelasArray2D<double> U, int ix, int q,
                                int i_eta) const -> double;
  [[nodiscard]] auto basis_eval(AthelasArray1D<double> U, int ix,
                                int i_eta) const -> double;
  void compute_mass_matrix(const AthelasArray3D<double> uPF,
                           const GridStructure *grid);

  [[nodiscard]] auto get_phi(int ix, int i_eta, int k) const -> double;
  [[nodiscard]] auto phi() const noexcept -> AthelasArray3D<double>;
  [[nodiscard]] auto get_d_phi(int ix, int i_eta, int k) const -> double;
  [[nodiscard]] auto dphi() const noexcept -> AthelasArray3D<double>;
  [[nodiscard]] auto get_mass_matrix(int ix, int k) const -> double;

  [[nodiscard]] auto get_order() const noexcept -> int;

  // L2 projection from nodal to modal representation
  void project_nodal_to_modal(
      AthelasArray3D<double> uCF, AthelasArray3D<double> uPF,
      GridStructure *grid, int q, int ix,
      const std::function<double(double, int, int)> &nodal_func) const;

  // L2 projection from nodal to modal representation for all cells
  void project_nodal_to_modal_all_cells(
      AthelasArray3D<double> uCF, AthelasArray3D<double> uPF,
      GridStructure *grid, int q,
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

  AthelasArray2D<double> mass_matrix_;

  AthelasArray3D<double> phi_;
  AthelasArray3D<double> dphi_;

  double (*func_)(const int n, const double x, const double x_c);
  double (*dfunc_)(const int n, const double x, double const x_c);
};

} // namespace athelas::basis
