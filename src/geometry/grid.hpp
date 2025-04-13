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
  [[nodiscard]] auto NodeCoordinate( int iC, int iN ) const -> Real;
  [[nodiscard]] auto Get_Centers( int iC ) const -> Real;
  [[nodiscard]] auto Get_Widths( int iC ) const -> Real;
  [[nodiscard]] auto Get_Nodes( int nN ) const -> Real;
  [[nodiscard]] auto Get_Weights( int nN ) const -> Real;
  [[nodiscard]] auto Get_Mass( int iX ) const -> Real;
  [[nodiscard]] auto Get_CenterOfMass( int iX ) const -> Real;
  [[nodiscard]] auto Get_xL( ) const -> Real;
  [[nodiscard]] auto Get_xR( ) const -> Real;
  [[nodiscard]] auto Get_SqrtGm( Real X ) const -> Real;
  [[nodiscard]] auto Get_LeftInterface( int iX ) const -> Real;

  [[nodiscard]] auto DoGeometry( ) const noexcept -> bool;

  [[nodiscard]] auto Get_Guard( ) const noexcept -> int;
  [[nodiscard]] auto Get_ilo( ) const noexcept -> int;
  [[nodiscard]] auto Get_ihi( ) const noexcept -> int;
  [[nodiscard]] auto Get_nNodes( ) const noexcept -> int;
  [[nodiscard]] auto Get_nElements( ) const noexcept -> int;

  void CreateGrid( ) const;
  void UpdateGrid( View1D<Real> SData );
  void ComputeMass( View3D<Real> uPF );
  void ComputeCenterOfMass( View3D<Real> uPF );
  void ComputeCenterOfMass_Radius( View3D<Real> uPF );

  auto operator( )( int i, int j ) -> Real&;
  auto operator( )( int i, int j ) const -> Real;

 private:
  int nElements;
  int nNodes;
  int nGhost;
  int mSize;

  Real xL;
  Real xR;

  geometry::Geometry Geometry;

  View1D<Real> Nodes{ };
  View1D<Real> Weights{ };

  View1D<Real> Centers{ };
  View1D<Real> Widths{ };
  View1D<Real> X_L{ }; // left interface coordinate

  View1D<Real> Mass{ };
  View1D<Real> CenterOfMass{ };

  View2D<Real> Grid{ };
};

#endif // GRID_HPP_
