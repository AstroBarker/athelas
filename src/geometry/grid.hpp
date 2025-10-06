#pragma once

#include "kokkos_types.hpp"
#include "pgen/problem_in.hpp"

namespace athelas {

enum class Geometry { Planar, Spherical };

enum class Domain { Interior, Entire };

class GridStructure {
 public:
  explicit GridStructure(const ProblemIn *pin);
  GridStructure() = default;
  KOKKOS_FUNCTION
  [[nodiscard]] auto node_coordinate(int iC, int q) const -> double;
  KOKKOS_FUNCTION
  [[nodiscard]] auto centers(int iC) const -> double;
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

  void create_grid(const ProblemIn *pin);
  void create_uniform_grid();
  void create_log_grid();

  KOKKOS_FUNCTION
  void update_grid(AthelasArray1D<double> SData);
  KOKKOS_FUNCTION
  void compute_mass(AthelasArray3D<double> uPF);
  KOKKOS_FUNCTION
  void compute_mass_r(AthelasArray3D<double> uPF);
  KOKKOS_FUNCTION
  auto enclosed_mass(int ix, int iN) const noexcept -> double;
  KOKKOS_FUNCTION
  void compute_center_of_mass(AthelasArray3D<double> uPF);
  KOKKOS_FUNCTION
  void compute_center_of_mass_radius(AthelasArray3D<double> uPF);

  [[nodiscard]] auto widths() const -> AthelasArray1D<double>;
  [[nodiscard]] auto mass() const -> AthelasArray1D<double>;
  [[nodiscard]] auto centers() const -> AthelasArray1D<double>;
  [[nodiscard]] auto centers() -> AthelasArray1D<double>;
  [[nodiscard]] auto nodal_grid() -> AthelasArray2D<double>;
  [[nodiscard]] auto nodal_grid() const -> AthelasArray2D<double>;

  // domain
  template <Domain D>
  [[nodiscard]] auto domain() const noexcept -> std::pair<int, int> {
    if constexpr (D == Domain::Interior) {
      return {1, nElements_};
    } else if constexpr (D == Domain::Entire) {
      return {0, nElements_ + 1};
    }
  }

  KOKKOS_FUNCTION
  auto operator()(int i, int j) -> double &;
  KOKKOS_FUNCTION
  auto operator()(int i, int j) const -> double;

 private:
  int nElements_;
  int nNodes_;
  int mSize_;

  double xL_;
  double xR_;

  Geometry geometry_;
  std::string grid_type_; // uniform or logarithmic

  AthelasArray1D<double> nodes_;
  AthelasArray1D<double> weights_;

  AthelasArray1D<double> centers_;
  AthelasArray1D<double> widths_;
  AthelasArray1D<double> x_l_; // left interface coordinate

  AthelasArray1D<double> mass_; // cell mass
  AthelasArray2D<double> mass_r_; // enclosed mass
  AthelasArray1D<double> center_of_mass_;

  AthelasArray2D<double> grid_;
};

} // namespace athelas
