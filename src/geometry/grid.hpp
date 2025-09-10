#pragma once
/**
 * @file grid.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Class for holding the spatial grid.
 *
 * @details This class GridStructure holds key pieces of the grid:
 *          - nx
 *          - nnodes
 *          - weights
 *
 *          For a loop over real zones, loop from ilo to ihi (inclusive).
 *          ilo = nGhost
 *          ihi = nElements - nGhost + 1
 */

#include "abstractions.hpp"
#include "geometry.hpp"
#include "problem_in.hpp"

class GridStructure {
 public:
  explicit GridStructure(const ProblemIn* pin);
  GridStructure() = default;
  KOKKOS_FUNCTION
  [[nodiscard]] auto node_coordinate(int iC, int iN) const -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_centers(int iC) const -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_widths(int iC) const -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_nodes(int nN) const -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_weights(int nN) const -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_mass(int ix) const -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_center_of_mass(int ix) const -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_x_l() const noexcept -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_x_r() const noexcept -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_sqrt_gm(double X) const -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_left_interface(int ix) const -> double;

  KOKKOS_FUNCTION
  [[nodiscard]] auto do_geometry() const noexcept -> bool;

  KOKKOS_FUNCTION
  [[nodiscard]] static auto get_ilo() noexcept -> int;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_ihi() const noexcept -> int;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_n_nodes() const noexcept -> int;
  KOKKOS_FUNCTION
  [[nodiscard]] auto get_n_elements() const noexcept -> int;

  void create_grid(const ProblemIn* pin);
  void create_uniform_grid();
  void create_log_grid();

  KOKKOS_FUNCTION
  void update_grid(View1D<double> SData);
  KOKKOS_FUNCTION
  void compute_mass(View3D<double> uPF);
  KOKKOS_FUNCTION
  void compute_mass_r(View3D<double> uPF);
  KOKKOS_FUNCTION
  auto enclosed_mass(int ix, int iN) const noexcept -> double;
  KOKKOS_FUNCTION
  void compute_center_of_mass(View3D<double> uPF);
  KOKKOS_FUNCTION
  void compute_center_of_mass_radius(View3D<double> uPF);

  [[nodiscard]] auto widths() -> View1D<double>;
  [[nodiscard]] auto centers() -> View1D<double>;
  [[nodiscard]] auto nodal_grid() -> View2D<double>;

  KOKKOS_FUNCTION
  auto operator()(int i, int j) -> double&;
  KOKKOS_FUNCTION
  auto operator()(int i, int j) const -> double;

 private:
  int nElements_;
  int nNodes_;
  int mSize_;

  double xL_;
  double xR_;

  geometry::Geometry geometry_;
  std::string grid_type_; // uniform or logarithmic

  View1D<double> nodes_{};
  View1D<double> weights_{};

  View1D<double> centers_{};
  View1D<double> widths_{};
  View1D<double> x_l_{}; // left interface coordinate

  View1D<double> mass_{}; // cell mass
  View2D<double> mass_r_{}; // enclosed mass
  View1D<double> center_of_mass_{};

  View2D<double> grid_{};
};
