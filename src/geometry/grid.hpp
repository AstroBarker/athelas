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
  [[nodiscard]] auto node_coordinate(int iC, int iN) const -> double;
  [[nodiscard]] auto get_centers(int iC) const -> double;
  [[nodiscard]] auto get_widths(int iC) const -> double;
  [[nodiscard]] auto get_nodes(int nN) const -> double;
  [[nodiscard]] auto get_weights(int nN) const -> double;
  [[nodiscard]] auto get_mass(int iX) const -> double;
  [[nodiscard]] auto get_center_of_mass(int iX) const -> double;
  [[nodiscard]] auto get_x_l() const noexcept -> double;
  [[nodiscard]] auto get_x_r() const noexcept -> double;
  [[nodiscard]] auto get_sqrt_gm(double X) const -> double;
  [[nodiscard]] auto get_left_interface(int iX) const -> double;

  [[nodiscard]] auto do_geometry() const noexcept -> bool;

  [[nodiscard]] auto get_guard() const noexcept -> int;
  [[nodiscard]] auto get_ilo() const noexcept -> int;
  [[nodiscard]] auto get_ihi() const noexcept -> int;
  [[nodiscard]] auto get_n_nodes() const noexcept -> int;
  [[nodiscard]] auto get_n_elements() const noexcept -> int;

  void create_grid();
  void update_grid(View1D<double> SData);
  void compute_mass(View3D<double> uPF);
  void compute_center_of_mass(View3D<double> uPF);
  void compute_center_of_mass_radius(View3D<double> uPF);

  auto operator()(int i, int j) -> double&;
  auto operator()(int i, int j) const -> double;

 private:
  int nElements_;
  int nNodes_;
  int nGhost_;
  int mSize_;

  double xL_;
  double xR_;

  geometry::Geometry geometry_;

  View1D<double> nodes_{};
  View1D<double> weights_{};

  View1D<double> centers_{};
  View1D<double> widths_{};
  View1D<double> x_l_{}; // left interface coordinate

  View1D<double> mass_{};
  View1D<double> center_of_mass_{};

  View2D<double> grid_{};
};
