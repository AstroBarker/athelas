#pragma once

#include <concepts>

#include "geometry/grid.hpp"
#include "utils/abstractions.hpp"

// Concepts for physics packages
concept ExplicitPhysicsPackage = requires(
    T pkg, View3D<double> U, View3D<double> dU, const GridStructure& grid) {
  {pkg.update_explicit(U, dU, grid)}->std::same_as<void>;
  {pkg.name()}->std::convertible_to<std::string_view>;
  {pkg.max_timestep(U, grid)}->std::convertible_to<double>;
};

template <typename T>
concept ImplicitPhysicsPackage = requires(
    T pkg, View3D<double> U, View3D<double> dU, const GridStructure& grid) {
  {pkg.update_implicit(U, dU, grid)}->std::same_as<void>;
  {pkg.name()}->std::convertible_to<std::string_view>;
  {pkg.max_timestep(U, grid)}->std::convertible_to<double>;
};
