/**
 * @file grid.cpp
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
 *          ilo = nGhost_
 *          ihi = nElements_ - nGhost_ + 1
 */

#include <vector>

#include "grid.hpp"
#include "quadrature.hpp"

using namespace geometry;

GridStructure::GridStructure(const ProblemIn* pin)
    : nElements_(pin->param()->get<int>("problem.nx")),
      nNodes_(pin->param()->get<int>("fluid.nnodes")), mSize_(nElements_ + 2),
      xL_(pin->param()->get<double>("problem.xl")),
      xR_(pin->param()->get<double>("problem.xr")),
      geometry_(pin->param()->get<Geometry>("problem.geometry_model")),
      nodes_("Nodes", nNodes_), weights_("weights_", nNodes_),
      centers_("Cetners", mSize_), widths_("widths_", mSize_),
      x_l_("Left Interface", mSize_ + 1), mass_("Cell mass_", mSize_),
      mass_r_("Enclosed mass", mSize_, nNodes_),
      center_of_mass_("Center of mass_", mSize_),
      grid_("Grid", mSize_, nNodes_) {
  std::vector<double> tmp_nodes(nNodes_);
  std::vector<double> tmp_weights(nNodes_);

  for (int iN = 0; iN < nNodes_; iN++) {
    tmp_nodes[iN]   = 0.0;
    tmp_weights[iN] = 0.0;
  }

  quadrature::lg_quadrature(nNodes_, tmp_nodes, tmp_weights);

  for (int iN = 0; iN < nNodes_; iN++) {
    nodes_(iN)   = tmp_nodes[iN];
    weights_(iN) = tmp_weights[iN];
  }

  create_grid();
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
auto GridStructure::get_centers(int iC) const -> double { return centers_(iC); }

// Return cell width
KOKKOS_FUNCTION
auto GridStructure::get_widths(int iC) const -> double { return widths_(iC); }

// Return cell mass
KOKKOS_FUNCTION
auto GridStructure::get_mass(int iX) const -> double { return mass_(iX); }

// Return cell reference Center of mass_
KOKKOS_FUNCTION
auto GridStructure::get_center_of_mass(int iX) const -> double {
  return center_of_mass_(iX);
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
auto GridStructure::get_sqrt_gm(double X) const -> double {
  if (geometry_ == geometry::Spherical) {
    return X * X;
  }
  return 1.0;
}

// Accessor for x_l_
KOKKOS_FUNCTION
auto GridStructure::get_left_interface(int iX) const -> double {
  return x_l_(iX);
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

// Equidistant mesh
// TODO(astrobarker): We will need to replace centers_ here, right?
KOKKOS_FUNCTION
void GridStructure::create_grid() {

  const int ilo = 1; // first real zone
  const int ihi = nElements_; // last real zone

  for (int i = 0; i < nElements_ + 2; i++) {
    widths_(i) = (xR_ - xL_) / nElements_;
  }

  x_l_(1) = xL_;
  for (int iX = 2; iX < nElements_ + 2; iX++) {
    x_l_(iX) = x_l_(iX - 1) + widths_(iX - 1);
  }

  centers_(ilo) = xL_ + 0.5 * widths_(ilo);
  for (int i = ilo + 1; i <= ihi; i++) {
    centers_(i) = centers_(i - 1) + widths_(i - 1);
  }

  for (int i = ilo - 1; i >= 0; i--) {
    centers_(i) = centers_(i + 1) - widths_(i + 1);
  }
  for (int i = ihi + 1; i < nElements_ + 1 + 1; i++) {
    centers_(i) = centers_(i - 1) + widths_(i - 1);
  }

  for (int iC = ilo; iC <= ihi; iC++) {
    for (int iN = 0; iN < nNodes_; iN++) {
      grid_(iC, iN) = node_coordinate(iC, iN);
    }
  }
}

/**
 * Compute cell masses
 **/
KOKKOS_FUNCTION
void GridStructure::compute_mass(const View3D<double> uPF) {
  const int nNodes_ = get_n_nodes();
  const int ilo     = get_ilo();
  const int ihi     = get_ihi();

  double mass = 0.0;
  double X    = 0.0;

  for (int iX = ilo; iX <= ihi; iX++) {
    mass = 0.0;
    for (int iN = 0; iN < nNodes_; iN++) {
      X = node_coordinate(iX, iN);
      mass += weights_(iN) * get_sqrt_gm(X) * uPF(0, iX, iN);
    }
    mass *= widths_(iX);
    mass_(iX) = mass;
  }

  // Guard cells
  for (int iX = 0; iX < ilo; iX++) {
    mass_(ilo - 1 - iX) = mass_(ilo + iX);
    mass_(ihi + 1 + iX) = mass_(ihi - iX);
  }
}

/**
 * Compute enclosed masses
 **/
KOKKOS_FUNCTION
void GridStructure::compute_mass_r(const View3D<double> uPF) {
  const int nNodes_ = get_n_nodes();
  const int ilo     = get_ilo();
  const int ihi     = get_ihi();

  double mass = 0.0;
  double X    = 0.0;

  const double geom_fac = (do_geometry()) ? 4.0 * constants::PI : 1.0;

  mass = 0.0;
  for (int iX = ilo; iX <= ihi; ++iX) {
    for (int iN = 0; iN < nNodes_; ++iN) {
      X = node_coordinate(iX, iN);
      mass += weights_(iN) * get_sqrt_gm(X) * uPF(0, iX, iN);
      mass_r_(iX, iN) = mass * widths_(iX) * geom_fac;
    }
  }
}

KOKKOS_FUNCTION
auto GridStructure::enclosed_mass(const int iX, const int iN) const noexcept
    -> double {
  return mass_r_(iX, iN);
}

/**
 * Compute cell centers of masses reference coordinates
 **/
KOKKOS_FUNCTION
void GridStructure::compute_center_of_mass(const View3D<double> uPF) {
  const int nNodes_ = get_n_nodes();
  const int ilo     = get_ilo();
  const int ihi     = get_ihi();

  double com = 0.0;
  double X   = 0.0;

  for (int iX = ilo; iX <= ihi; iX++) {
    com = 0.0;
    for (int iN = 0; iN < nNodes_; iN++) {
      X = node_coordinate(iX, iN);
      com += nodes_(iN) * weights_(iN) * get_sqrt_gm(X) * uPF(0, iX, iN);
    }
    com *= widths_(iX);
    center_of_mass_(iX) = com / mass_(iX);
  }

  // Guard cells
  for (int iX = 0; iX < ilo; iX++) {
    center_of_mass_(ilo - 1 - iX) = center_of_mass_(ilo + iX);
    center_of_mass_(ihi + 1 + iX) = center_of_mass_(ihi - iX);
  }
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
      KOKKOS_CLASS_LAMBDA(int iX) {
        x_l_(iX)     = SData(iX);
        widths_(iX)  = SData(iX + 1) - SData(iX);
        centers_(iX) = 0.5 * (SData(iX + 1) + SData(iX));
      });

  Kokkos::parallel_for(
      "Grid Update 2", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(int iX) {
        for (int iN = 0; iN < nNodes_; iN++) {
          grid_(iX, iN) = node_coordinate(iX, iN);
        }
      });
}

// Access by (element, node)
KOKKOS_FUNCTION
auto GridStructure::operator()(int i, int j) -> double& { return grid_(i, j); }

KOKKOS_FUNCTION
auto GridStructure::operator()(int i, int j) const -> double {
  return grid_(i, j);
}
