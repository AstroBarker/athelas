/**
 * @file grid.cpp
 * --------------
 *
 * @brief Class for holding the spatial grid.
 *
 * @details This class GridStructure holds key pieces of the grid:
 *          - nx
 *          - nnodes
 *          - weights
 *
 *          For a loop over real zones, loop from ilo to ihi (inclusive).
 *          ilo = nGhost_
 *          ihi = nElements_ - nGhost_ + 1
 */

#include <limits>
#include <vector>

#include "geometry/grid.hpp"
#include "quadrature/quadrature.hpp"
#include "utils/utilities.hpp"

using namespace geometry;

GridStructure::GridStructure(const ProblemIn *pin)
    : nElements_(pin->param()->get<int>("problem.nx")),
      nNodes_(pin->param()->get<int>("fluid.nnodes")), mSize_(nElements_ + 2),
      xL_(pin->param()->get<double>("problem.xl")),
      xR_(pin->param()->get<double>("problem.xr")),
      geometry_(pin->param()->get<Geometry>("problem.geometry_model")),
      grid_type_(pin->param()->get<std::string>("problem.grid_type")),
      nodes_("Nodes", nNodes_), weights_("weights_", nNodes_),
      centers_("Centers", mSize_), widths_("widths_", mSize_),
      x_l_("Left Interface", mSize_ + 1), mass_("Cell mass_", mSize_),
      mass_r_("Enclosed mass", mSize_, nNodes_),
      center_of_mass_("Center of mass_", mSize_),
      grid_("Grid", mSize_, nNodes_) {
  std::vector<double> tmp_nodes(nNodes_);
  std::vector<double> tmp_weights(nNodes_);

  for (int iN = 0; iN < nNodes_; iN++) {
    tmp_nodes[iN] = 0.0;
    tmp_weights[iN] = 0.0;
  }

  quadrature::lg_quadrature(nNodes_, tmp_nodes, tmp_weights);

  for (int iN = 0; iN < nNodes_; iN++) {
    nodes_(iN) = tmp_nodes[iN];
    weights_(iN) = tmp_weights[iN];
  }

  create_grid(pin);
}

// linear shape function on the reference element
KOKKOS_INLINE_FUNCTION
auto shape_function(const int interface, const double eta) -> double {
  if (interface == 0) {
    return 1.0 * (0.5 - eta);
  }
  if (interface == 1) {
    return 1.0 * (0.5 + eta);
  }
  return 0.0; // unreachable, but silences warnings
}

// Give physical grid coordinate from a node.
KOKKOS_FUNCTION
auto GridStructure::node_coordinate(int iC, int iN) const -> double {
  return x_l_(iC) * shape_function(0, nodes_(iN)) +
         x_l_(iC + 1) * shape_function(1, nodes_(iN));
}

// Return cell center
KOKKOS_FUNCTION
auto GridStructure::centers(int iC) const -> double { return centers_(iC); }

// Return cell width
KOKKOS_FUNCTION
auto GridStructure::get_widths(int iC) const -> double { return widths_(iC); }

// Return cell mass
KOKKOS_FUNCTION
auto GridStructure::get_mass(int ix) const -> double { return mass_(ix); }

// Return cell reference Center of mass_
KOKKOS_FUNCTION
auto GridStructure::get_center_of_mass(int ix) const -> double {
  return center_of_mass_(ix);
}

// Return given quadrature node
KOKKOS_FUNCTION
auto GridStructure::get_nodes(int nN) const -> double { return nodes_(nN); }

// Return given quadrature weight
KOKKOS_FUNCTION
auto GridStructure::get_weights(int nN) const -> double { return weights_(nN); }

// Accessor for xL
KOKKOS_FUNCTION
auto GridStructure::get_x_l() const noexcept -> double { return xL_; }

// Accessor for xR
KOKKOS_FUNCTION
auto GridStructure::get_x_r() const noexcept -> double { return xR_; }

// Accessor for SqrtGm
KOKKOS_FUNCTION
auto GridStructure::get_sqrt_gm(const double X) const -> double {
  if (geometry_ == geometry::Spherical) [[likely]] {
    return X * X;
  }
  return 1.0;
}

// Accessor for x_l_
KOKKOS_FUNCTION
auto GridStructure::get_left_interface(int ix) const -> double {
  return x_l_(ix);
}

// Return nNodes_
KOKKOS_FUNCTION
auto GridStructure::get_n_nodes() const noexcept -> int { return nNodes_; }

// Return nElements_
KOKKOS_FUNCTION
auto GridStructure::get_n_elements() const noexcept -> int {
  return nElements_;
}

// Return first physical zone
KOKKOS_FUNCTION
auto GridStructure::get_ilo() noexcept -> int { return 1; }

// Return last physical zone
KOKKOS_FUNCTION
auto GridStructure::get_ihi() const noexcept -> int { return nElements_; }

// Return true if in spherical symmetry
KOKKOS_FUNCTION
auto GridStructure::do_geometry() const noexcept -> bool {
  return geometry_ == geometry::Spherical;
}

// grid creation logic
void GridStructure::create_grid(const ProblemIn *pin) {
  if (utilities::to_lower(grid_type_) == "uniform") {
    create_uniform_grid();
  } else if (utilities::to_lower(grid_type_) == "logarithmic") {
    // Need to be careful of coordinates with log grid!
    if (xL_ < 0.0 || xR_ < 0.0) {
      THROW_ATHELAS_ERROR(
          "Negative coordinates are not supported with logarithmic gridding!");
    }
    if (xL_ == 0.0) {
      // We cannot have a boundary at 0 with a log grid. We replace the inner
      // boundary arbitrarily with 1.0e-4 * xR and warn the user to perhaps
      // consider asetting a better inner boundary.
      // This logic is not very general and might be made smarter.
      // It also is not an important edge case.
      const double new_xl = 1.0e-4 * xR_;
      std::stringstream ss;
      ss << std::scientific << std::setprecision(3) << new_xl;
      WARNING_ATHELAS("Logarithmic grid requestion with XL = 0.0. This does "
                      "not work. Setting xL = 10^-4 * xR = " +
                      ss.str() +
                      ". You might consider setting a more appropriate value.");

      // update outer boundary in params and in class
      auto &xl_pin = pin->param()->get_mutable_ref<double>("problem.xl");
      xl_pin = new_xl;
      xL_ = new_xl;
    }
    create_log_grid();
  } else {
    THROW_ATHELAS_ERROR("Unknown grid type '" + grid_type_ + "' provided!");
  }
}

/**
 * @brief uniform mesh
 */
void GridStructure::create_uniform_grid() {

  const int ilo = 1; // first real zone
  const int ihi = nElements_; // last real zone

  auto widths_h = Kokkos::create_mirror_view(widths_);
  auto centers_h = Kokkos::create_mirror_view(centers_);
  auto x_l_h = Kokkos::create_mirror_view(x_l_);

  for (int i = 0; i < nElements_ + 2; i++) {
    widths_h(i) = (xR_ - xL_) / nElements_;
  }

  x_l_h(1) = xL_;
  for (int ix = 2; ix < nElements_ + 2; ix++) {
    x_l_h(ix) = x_l_h(ix - 1) + widths_h(ix - 1);
  }

  centers_h(ilo) = xL_ + 0.5 * widths_h(ilo);
  for (int i = ilo + 1; i <= ihi; i++) {
    centers_h(i) = centers_h(i - 1) + widths_h(i - 1);
  }

  for (int i = ilo - 1; i >= 0; i--) {
    centers_h(i) = centers_h(i + 1) - widths_h(i + 1);
  }
  for (int i = ihi + 1; i < nElements_ + 1 + 1; i++) {
    centers_h(i) = centers_h(i - 1) + widths_h(i - 1);
  }

  // copy back to device mirrors
  Kokkos::deep_copy(widths_, widths_h);
  Kokkos::deep_copy(centers_, centers_h);
  Kokkos::deep_copy(x_l_, x_l_h);

  Kokkos::parallel_for(
      "Grid :: create_grid", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        for (int iN = 0; iN < nNodes_; iN++) {
          grid_(ix, iN) = node_coordinate(ix, iN);
        }
      });
}

/**
 * @brief logarithmic radial mesh generation
 *
 * Sets up logarithmic mesh with cell centers:
 * x_i = x_l * (x_r / x_l)^(i/(nx - 1))
 */
void GridStructure::create_log_grid() {

  const int ilo = 1; // first real zone
  const int ihi = nElements_; // last real zone

  auto widths_h = Kokkos::create_mirror_view(widths_);
  auto centers_h = Kokkos::create_mirror_view(centers_);
  auto x_l_h = Kokkos::create_mirror_view(x_l_);

  const double log_xl = std::log10(xL_);
  const double log_ratio = std::log10(utilities::ratio(xR_, xL_));
  const double dx = log_ratio / (nElements_ - 1);

  // Set up cell centers
  for (int i = ilo; i <= ihi; i++) {
    const double log_xi = log_xl + i * dx;
    centers_h(i) = std::pow(10.0, log_xi);
  }

  // Handle ghost cells
  for (int i = ilo - 1; i >= 0; i--) {
    centers_h(i) = centers_h(i + 1) - widths_h(i + 1);
  }
  for (int i = ihi + 1; i < nElements_ + 1 + 1; i++) {
    centers_h(i) = centers_h(i - 1) + widths_h(i - 1);
  }

  // Set up left edges
  x_l_h(1) = xL_;
  for (int i = 2; i < nElements_ + 2; i++) {
    x_l_h(i) = 0.5 * (centers_h(i - 1) + centers_h(i));
  }

  // Calculate remaining widths with geometric progression
  for (int i = 0; i < nElements_ + 2; i++) {
    widths_h(i) = x_l_h(i + 1) - x_l_h(i);
  }

  // copy back to device mirrors
  Kokkos::deep_copy(widths_, widths_h);
  Kokkos::deep_copy(centers_, centers_h);
  Kokkos::deep_copy(x_l_, x_l_h);

  Kokkos::parallel_for(
      "Grid :: create_log_grid", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        for (int iN = 0; iN < nNodes_; iN++) {
          grid_(ix, iN) = node_coordinate(ix, iN);
        }
      });
}

/**
 * Compute cell masses
 **/
KOKKOS_FUNCTION
void GridStructure::compute_mass(const View3D<double> uPF) {
  const int nNodes_ = get_n_nodes();
  const int ilo = get_ilo();
  const int ihi = get_ihi();

  Kokkos::parallel_for(
      "Grid :: compute_mass", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        double mass = 0.0;
        for (int iN = 0; iN < nNodes_; iN++) {
          const double X = node_coordinate(ix, iN);
          mass += weights_(iN) * get_sqrt_gm(X) * uPF(ix, iN, 0);
        }
        mass *= widths_(ix);
        mass_(ix) = mass;
      });

  // Guard cells
  Kokkos::parallel_for(
      "Grid :: compute_mass (ghost cells)", Kokkos::RangePolicy<>(0, ilo),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        mass_(ilo - 1 - ix) = mass_(ilo + ix);
        mass_(ihi + 1 + ix) = mass_(ihi - ix);
      });
}

/**
 * @brief Compute enclosed masses
 *
 * NOTE: This function is intended only to be called once, as it allocated.
 * If we find ourselves for any reason calling this repeatedly,
 * it should be refactored.
 **/
KOKKOS_FUNCTION
void GridStructure::compute_mass_r(const View3D<double> uPF) {
  const int nNodes_ = get_n_nodes();
  const int ilo = get_ilo();
  const int ihi = get_ihi();

  static const double geom_fac = (do_geometry()) ? 4.0 * constants::PI : 1.0;

  const int total_points = (ihi - ilo + 1) * nNodes_;
  View1D<double> mass_contrib("mass_contrib", total_points);
  View1D<double> cumulative_mass("cumulative_mass", total_points);

  // 1: Compute individual mass contributions in parallel
  Kokkos::parallel_for(
      "compute_mass_contributions", Kokkos::RangePolicy<>(0, total_points),
      KOKKOS_CLASS_LAMBDA(const int idx) {
        const int ix = ilo + idx / nNodes_;
        const int iN = idx % nNodes_;
        const double X = node_coordinate(ix, iN);
        mass_contrib(idx) = weights_(iN) * get_sqrt_gm(X) * uPF(ix, iN, 0);
      });

  // 2: Perform parallel inclusive scan (cumulative sum)
  Kokkos::parallel_scan(
      "compute_enclosed_mass", Kokkos::RangePolicy<>(0, total_points),
      KOKKOS_LAMBDA(const int idx, double &partial_sum, const bool is_final) {
        partial_sum += mass_contrib(idx);
        if (is_final) {
          cumulative_mass(idx) = partial_sum;
        }
      });

  // 3: sort into mass_r_
  Kokkos::parallel_for(
      "store_enclosed_mass", Kokkos::RangePolicy<>(0, total_points),
      KOKKOS_CLASS_LAMBDA(const int idx) {
        const int ix = ilo + idx / nNodes_;
        const int iN = idx % nNodes_;
        mass_r_(ix, iN) = cumulative_mass(idx) * widths_(ix) * geom_fac;
      });

  // Get total mass
  // auto h_cumulative =
  // Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cumulative_mass);
  // mass = h_cumulative(total_points - 1);
}

KOKKOS_FUNCTION
auto GridStructure::enclosed_mass(const int ix, const int iN) const noexcept
    -> double {
  return mass_r_(ix, iN);
}

/**
 * Compute cell centers of masses reference coordinates
 **/
KOKKOS_FUNCTION
void GridStructure::compute_center_of_mass(const View3D<double> uPF) {
  const int nNodes_ = get_n_nodes();
  const int ilo = get_ilo();
  const int ihi = get_ihi();

  Kokkos::parallel_for(
      "Grid :: compute_center_of_mass", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        double com = 0.0;

        for (int iN = 0; iN < nNodes_; iN++) {
          const double X = node_coordinate(ix, iN);
          com += nodes_(iN) * weights_(iN) * get_sqrt_gm(X) * uPF(ix, iN, 0);
        }

        com *= widths_(ix);
        center_of_mass_(ix) = com / mass_(ix);
      });

  // Guard cells
  Kokkos::parallel_for(
      "Grid :: compute_center_of_mass (ghost cells)",
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_CLASS_LAMBDA(const int ix) {
        center_of_mass_(ilo - 1 - ix) = center_of_mass_(ilo + ix);
        center_of_mass_(ihi + 1 + ix) = center_of_mass_(ihi - ix);
      });
}

/**
 * Update grid coordinates using interface velocities.
 **/
KOKKOS_FUNCTION
void GridStructure::update_grid(const View1D<double> SData) {

  const int ilo = get_ilo();
  const int ihi = get_ihi();

  Kokkos::parallel_for(
      "Grid Update 1", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(int ix) {
        x_l_(ix) = SData(ix);
        widths_(ix) = SData(ix + 1) - SData(ix);
        centers_(ix) = 0.5 * (SData(ix + 1) + SData(ix));
      });

  Kokkos::parallel_for(
      "Grid Update 2", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes_; iN++) {
          grid_(ix, iN) = node_coordinate(ix, iN);
        }
      });
}

// Access by (element, node)
KOKKOS_FUNCTION
auto GridStructure::operator()(int i, int j) -> double & { return grid_(i, j); }

KOKKOS_FUNCTION
auto GridStructure::operator()(int i, int j) const -> double {
  return grid_(i, j);
}

[[nodiscard]] auto GridStructure::widths() const -> View1D<double> {
  return widths_;
}
[[nodiscard]] auto GridStructure::mass() const -> View1D<double> {
  return mass_;
}
[[nodiscard]] auto GridStructure::centers() const -> View1D<double> {
  return centers_;
}
[[nodiscard]] auto GridStructure::centers() -> View1D<double> {
  return centers_;
}
[[nodiscard]] auto GridStructure::nodal_grid() -> View2D<double> {
  return grid_;
}
[[nodiscard]] auto GridStructure::nodal_grid() const -> View2D<double> {
  return grid_;
}
