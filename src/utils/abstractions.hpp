#pragma once
/**
 * @file abstractions.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Provides useful definitions.
 */

#include "Kokkos_Core.hpp"

using DevMemSpace     = Kokkos::DefaultExecutionSpace::memory_space;
using HostMemSpace    = Kokkos::HostSpace;
using DevExecSpace    = Kokkos::DefaultExecutionSpace;
using ScratchMemSpace = DevExecSpace::scratch_memory_space;

using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemUnmanaged  = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

template <typename T>
using ScratchPad1D = Kokkos::View<T*, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad2D = Kokkos::View<T**, ScratchMemSpace, MemUnmanaged>;

template <typename T>
using View4D = Kokkos::View<T****>;

template <typename T>
using View3D = Kokkos::View<T***>;

template <typename T>
using View2D = Kokkos::View<T**>;

template <typename T>
using View1D = Kokkos::View<T*>;

template <typename T>
using HostView1D = Kokkos::View<T*, Kokkos::HostSpace>;

/* Where to put this? */
namespace poly_basis {
enum poly_basis { legendre, taylor };
} // namespace poly_basis
//
struct TimeStepInfo {
  double t;
  double dt;
  double dt_a; // dt * tableau coefficient
  int stage;
};
enum class GravityModel { Constant, Spherical };
