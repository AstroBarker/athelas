/**
 * @file abstractions.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Provides useful definitions.
 */

#pragma once

#include "Kokkos_Core.hpp"

#include "basic_types.hpp"

namespace athelas {

using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostMemSpace = Kokkos::HostSpace;
using DevExecSpace = Kokkos::DefaultExecutionSpace;
using ScratchMemSpace = DevExecSpace::scratch_memory_space;

using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

template <typename T>
using View4D = Kokkos::View<T ****>;

template <typename T>
using View3D = Kokkos::View<T ***>;

template <typename T>
using View2D = Kokkos::View<T **>;

template <typename T>
using View1D = Kokkos::View<T *>;

template <typename T>
using HostView1D = Kokkos::View<T *, Kokkos::HostSpace>;

namespace custom_reductions { // namespace helps with name resolution in
                              // reduction identity
template <class ScalarType, int N>
struct array_type {
  ScalarType data[N];

  KOKKOS_INLINE_FUNCTION // Default constructor - Initialize to 0's
  array_type() {
    for (int i = 0; i < N; i++) {
      data[i] = 0;
    }
  }
  KOKKOS_INLINE_FUNCTION // Copy Constructor
  array_type(const array_type &rhs) {
    for (int i = 0; i < N; i++) {
      data[i] = rhs.data[i];
    }
  }
  KOKKOS_INLINE_FUNCTION // add operator
      array_type &
      operator+=(const array_type &src) {
    for (int i = 0; i < N; i++) {
      data[i] += src.data[i];
    }
    return *this;
  }
};
using ValueType = array_type<double, 4>;

struct ArgMax {
  int index;
  double value;

  KOKKOS_INLINE_FUNCTION
  ArgMax() : index(-1), value(-Kokkos::reduction_identity<double>::min()) {}

  KOKKOS_INLINE_FUNCTION
  ArgMax(int idx, double val) : index(idx), value(val) {}

  KOKKOS_INLINE_FUNCTION
  ArgMax &operator=(const ArgMax &other) {
    index = other.index;
    value = other.value;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  auto operator>(const ArgMax &rhs) const -> bool {
    return this->value > rhs.value;
  }
};
} // namespace custom_reductions

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

} // namespace athelas
  //
namespace Kokkos { // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<athelas::custom_reductions::ValueType> {
  KOKKOS_FORCEINLINE_FUNCTION static athelas::custom_reductions::ValueType
  sum() {
    return athelas::custom_reductions::ValueType();
  }
};
template <>
struct reduction_identity<athelas::custom_reductions::ArgMax> {
  KOKKOS_FORCEINLINE_FUNCTION
  static athelas::custom_reductions::ArgMax max() {
    return athelas::custom_reductions::ArgMax(
        -1, -Kokkos::reduction_identity<double>::max());
  }
};
} // namespace Kokkos
