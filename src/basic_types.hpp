#pragma once

#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <Kokkos_Core.hpp>

namespace athelas {

struct IndexRange {

  IndexRange() = default;
  explicit IndexRange(const int n) : e(n - 1) {}
  IndexRange(const int start, const int end) : s(start), e(end) {}
  explicit IndexRange(std::pair<int, int> domain)
      : s(domain.first), e(domain.second) {}

  int s = 0; /// Starting Index (inclusive)
  int e = 0; /// Ending Index (inclusive)
  [[nodiscard]] auto size() const -> int { return e - s + 1; }
  explicit operator std::pair<int, int>() const { return {s, e}; }
};

/**
 * @struct TimeStepInfo
 * @brief holds information related to a timestep
 */
struct TimeStepInfo {
  double t;
  double dt;
  double dt_a; // dt * tableau coefficient
  int stage;
};

enum class poly_basis { legendre, taylor };

enum class GravityModel { Constant, Spherical };

template <typename T>
using Dictionary = std::unordered_map<std::string, T>;

template <typename T>
using Triple_t = std::tuple<T, T, T>;
} // namespace athelas
