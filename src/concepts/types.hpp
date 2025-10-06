#pragma once

namespace athelas {

template <typename F>
concept Functor = requires(F f) {
  { &F::operator() };
};

} // namespace athelas
