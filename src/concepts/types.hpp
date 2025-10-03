#pragma once

template <typename F>
concept Functor = requires(F f) {
  { &F::operator() };
};
