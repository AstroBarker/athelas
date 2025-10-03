#pragma once

#include <string>
#include <tuple>
#include <utility>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "utils/type_list.hpp"

/**
 * The code here is largely borrowed from Parthenon
 * https://github.com/parthenon-hpc-lab/parthenon
 */

namespace athelas {

template <class... Ts>
struct Indexer {
  KOKKOS_INLINE_FUNCTION
  Indexer() : start{}, N_{} {};

  [[nodiscard]] auto get_ranges_string() const -> std::string {
    auto end = End();
    std::string out;
    for (int i = 0; i < sizeof...(Ts); ++i) {
      out +=
          "[ " + std::to_string(start[i]) + ", " + std::to_string(end[i]) + "]";
    }
    return out;
  }

  KOKKOS_INLINE_FUNCTION
  explicit Indexer(std::pair<Ts, Ts>... Ns)
      : start{Ns.first...},
        N_{GetFactors({(Ns.second - Ns.first + 1)...},
                      std::make_index_sequence<sizeof...(Ts)>())} {}

  template <class... IndRngs>
  KOKKOS_INLINE_FUNCTION explicit Indexer(IndRngs... Ns)
      : start{Ns.s...},
        N_{GetFactors({(Ns.e - Ns.s + 1)...},
                      std::make_index_sequence<sizeof...(Ts)>())} {}

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto size() const -> std::size_t {
    return N_[0];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto operator()(int idx) const -> std::tuple<Ts...> {
    return GetIndicesImpl(idx, std::make_index_sequence<sizeof...(Ts)>());
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto get_flat_idx(Ts... ts) const -> std::size_t {
    return GetFlatIndexImpl(ts..., std::make_index_sequence<sizeof...(Ts)>());
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto get_idx_array(int idx) const {
    return GetIndicesKArrayImpl(idx, std::make_index_sequence<sizeof...(Ts)>());
  }

  template <std::size_t I>
  KOKKOS_FORCEINLINE_FUNCTION auto start_idx() const {
    return start[I];
  }

  template <std::size_t I>
  KOKKOS_FORCEINLINE_FUNCTION auto end_idx() const {
    const std::size_t ni = N_[I] / get_n<I>();
    int end = ni + start[I] - 1;
    return end;
  }

  KOKKOS_FORCEINLINE_FUNCTION auto End() const {
    return End_impl(std::make_index_sequence<sizeof...(Ts)>());
  }

  static const constexpr std::size_t rank = sizeof...(Ts);

 protected:
  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::tuple<Ts...>
  GetIndicesImpl(int idx, std::index_sequence<Is...>) const {
    std::tuple<Ts...> idxs;
    (
        [&] {
          std::get<Is>(idxs) = idx / get_n<Is>();
          idx -= std::get<Is>(idxs) * get_n<Is>();
          std::get<Is>(idxs) += start[Is];
        }(),
        ...);
    return idxs;
  }

  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<int, sizeof...(Ts)>
  GetIndicesKArrayImpl(int idx, std::index_sequence<Is...>) const {
    Kokkos::Array<int, sizeof...(Ts)> indices;
    (
        [&] {
          indices[Is] = idx / get_n<Is>();
          idx -= indices[Is] * get_n<Is>();
          indices[Is] += start[Is];
        }(),
        ...);
    return indices;
  }

  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::size_t
  GetFlatIndexImpl(Ts... idxs, std::index_sequence<Is...>) const {
    std::size_t out{0};
    (
        [&] {
          idxs -= start[Is];
          out += idxs * get_n<Is>();
        }(),
        ...);
    return out;
  }

  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Array<int, sizeof...(Ts)>
  GetFactors(Kokkos::Array<int, sizeof...(Ts)> Nt, std::index_sequence<Is...>) {
    Kokkos::Array<int, sizeof...(Ts)> N;
    std::size_t cur = 1;
    (
        [&] {
          constexpr std::size_t idx = sizeof...(Ts) - (Is + 1);
          cur *= Nt[idx];
          N[idx] = cur;
        }(),
        ...);
    return N;
  }

  Kokkos::Array<int, sizeof...(Ts)> start;

 private:
  template <std::size_t I>
  KOKKOS_FORCEINLINE_FUNCTION const auto get_n() const {
    if constexpr (I == sizeof...(Ts) - 1) {
      return 1;
    }

    return N_[I + 1];
  }

  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION auto End_impl(std::index_sequence<Is...>) const {
    Kokkos::Array<int, sizeof...(Ts)> end;
    ([&] { end[Is] = end_idx<Is>(); }(), ...);
    return end;
  }

  Kokkos::Array<int, sizeof...(Ts)> N_;
};

template <class... Ts>
struct IndexRanger {
  KOKKOS_INLINE_FUNCTION
  IndexRanger() : N{}, _size{} {};

  KOKKOS_INLINE_FUNCTION
  explicit IndexRanger(Ts... IdrsA) {}

  Kokkos::Array<IndexRange, sizeof...(Ts)> N;
  std::size_t _size;
};

template <>
struct Indexer<> {
  // this is a dummy and shouldn't ever actually get used to index an array
  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<int, 1>
  get_idx_array(int idx) const {
    return {-1};
  }
};

using Indexer1D = Indexer<int>;
using Indexer2D = Indexer<int, int>;
using Indexer3D = Indexer<int, int, int>;
using Indexer4D = Indexer<int, int, int, int>;

template <class... Ts>
KOKKOS_FORCEINLINE_FUNCTION auto
make_indexer(const std::pair<Ts, Ts> &...ranges) {
  return Indexer<Ts...>(ranges...);
}

template <std::size_t NIdx, class... Ts, std::size_t... Is>
KOKKOS_FORCEINLINE_FUNCTION auto
make_indexer(TypeList<Ts...>, Kokkos::Array<IndexRange, NIdx> bounds_arr,
             std::integer_sequence<std::size_t, Is...>) {
  return Indexer<Ts...>(bounds_arr[Is]...);
}

template <std::size_t NIdx>
KOKKOS_FORCEINLINE_FUNCTION auto
make_indexer(Kokkos::Array<IndexRange, NIdx> bounds_arr) {
  return make_indexer(list_of_type_t<NIdx, IndexRange>(), bounds_arr,
                      std::make_index_sequence<NIdx>());
}

} // namespace athelas
