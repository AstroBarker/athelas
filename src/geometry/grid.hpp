#ifndef GRID_HPP_
#define GRID_HPP_
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

#include <algorithm> // std::copy
#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "error.hpp"
#include "geometry.hpp"
#include "problem_in.hpp"
#include "quadrature.hpp"

class GridStructure {
 public:
  explicit GridStructure( const ProblemIn* pin );
  [[nodiscard]] auto node_coordinate( int iC, int iN ) const -> Real;
  [[nodiscard]] auto get_centers( int iC ) const -> Real;
  [[nodiscard]] auto get_widths( int iC ) const -> Real;
  [[nodiscard]] auto get_nodes( int nN ) const -> Real;
  [[nodiscard]] auto get_weights( int nN ) const -> Real;
  [[nodiscard]] auto get_mass( int iX ) const -> Real;
  [[nodiscard]] auto get_center_of_mass( int iX ) const -> Real;
  [[nodiscard]] auto get_x_l( ) const noexcept -> Real;
  [[nodiscard]] auto get_x_r( ) const noexcept -> Real;
  [[nodiscard]] auto get_sqrt_gm( Real X ) const -> Real;
  [[nodiscard]] auto get_left_interface( int iX ) const -> Real;

  [[nodiscard]] auto do_geometry( ) const noexcept -> bool;

  [[nodiscard]] auto get_guard( ) const noexcept -> int;
  [[nodiscard]] auto get_ilo( ) const noexcept -> int;
  [[nodiscard]] auto get_ihi( ) const noexcept -> int;
  [[nodiscard]] auto get_n_nodes( ) const noexcept -> int;
  [[nodiscard]] auto get_n_elements( ) const noexcept -> int;

  void create_grid( );
  void update_grid( View1D<Real> SData );
  void compute_mass( View3D<Real> uPF );
  void compute_center_of_mass( View3D<Real> uPF );
  void compute_center_of_mass_radius( View3D<Real> uPF );

  auto operator( )( int i, int j ) -> Real&;
  auto operator( )( int i, int j ) const -> Real;

 private:
  int nElements_;
  int nNodes_;
  int nGhost_;
  int mSize_;

  Real xL_;
  Real xR_;

  geometry::Geometry geometry_;

  View1D<Real> nodes_{ };
  View1D<Real> weights_{ };

  View1D<Real> centers_{ };
  View1D<Real> widths_{ };
  View1D<Real> x_l_{ }; // left interface coordinate

  View1D<Real> mass_{ };
  View1D<Real> center_of_mass_{ };

  View2D<Real> grid_{ };
};

#endif // GRID_HPP_
