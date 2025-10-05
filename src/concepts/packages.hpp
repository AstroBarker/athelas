#pragma once

#include <concepts>

#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

namespace athelas {

// Concepts for package validation
template <typename T>
concept ExplicitPackage =
    requires(T &pkg, const State *const state, State *state_derived,
             View3D<double> uCF, View3D<double> dU, const GridStructure &grid,
             const TimeStepInfo &dt_info) {
      { pkg.update_explicit(state, dU, grid, dt_info) } -> std::same_as<void>;
      { pkg.min_timestep(state, grid, dt_info) } -> std::convertible_to<double>;
      { pkg.name() } -> std::convertible_to<std::string_view>;
      { pkg.is_active() } -> std::convertible_to<bool>;
      { pkg.fill_derived(state_derived, grid, dt_info) } -> std::same_as<void>;
    };

template <typename T>
concept ImplicitPackage =
    requires(T &pkg, const State *const state, State *state_derived,
             View3D<double> uCF, View3D<double> dU, const GridStructure &grid,
             const TimeStepInfo &dt_info) {
      { pkg.update_implicit(state, dU, grid, dt_info) } -> std::same_as<void>;
      { pkg.min_timestep(state, grid, dt_info) } -> std::convertible_to<double>;
      { pkg.name() } -> std::convertible_to<std::string_view>;
      { pkg.is_active() } -> std::convertible_to<bool>;
      { pkg.fill_derived(state_derived, grid, dt_info) } -> std::same_as<void>;
    };

template <typename T>
concept IMEXPackage =
    requires(T &pkg, const State *const state, State *state_derived,
             View3D<double> uCF, View3D<double> dU, const GridStructure &grid,
             const TimeStepInfo &dt_info) {
      { pkg.update_explicit(state, dU, grid, dt_info) } -> std::same_as<void>;
      { pkg.update_implicit(state, dU, grid, dt_info) } -> std::same_as<void>;
      {
        pkg.update_implicit_iterative(state, dU, grid, dt_info)
      } -> std::same_as<void>;
      { pkg.min_timestep(state, grid, dt_info) } -> std::convertible_to<double>;
      { pkg.name() } -> std::convertible_to<std::string_view>;
      { pkg.is_active() } -> std::convertible_to<bool>;
      { pkg.fill_derived(state_derived, grid, dt_info) } -> std::same_as<void>;
    };

template <typename T>
concept PhysicsPackage =
    ExplicitPackage<T> || ImplicitPackage<T> || IMEXPackage<T>;

// Type traits to detect package capabilities
template <typename T>
constexpr bool has_explicit_update_v =
    requires(T &pkg, const State *const state, View3D<double> dU,
             const GridStructure &grid, const TimeStepInfo &dt_info) {
      pkg.update_explicit(state, dU, grid, dt_info);
    };

template <typename T>
constexpr bool has_implicit_update_v =
    requires(T &pkg, const State *const state, View3D<double> dU,
             const GridStructure &grid, const TimeStepInfo &dt_info) {
      pkg.update_implicit(state, dU, grid, dt_info);
    };

} // namespace athelas
