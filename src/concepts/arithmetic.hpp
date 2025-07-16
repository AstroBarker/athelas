#pragma once

#include <concepts>

// Define a concept that ensures subtraction is valid
template <typename T>
concept Subtractable = requires( T a, T b ) {
  { a - b }->std::convertible_to<T>;
};
