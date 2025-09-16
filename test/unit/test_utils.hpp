#pragma once
/**
 * Utilities for testing
 * Contains:
 * SoftEqual
 **/

#include <cmath>
#include <iostream>
#include <print>

#include "io/parser.hpp"

/**
 * Test for near machine precision
 **/
inline bool soft_equal(const double &val, const double &ref,
                       const double tol = 1.0e-8) {
  if (std::abs(val - ref) < tol * std::abs(ref) + tol) {
    return true;
  } else {
    return false;
  }
}

// Utility function to print parse results
inline void print_parser_data(const Parser::ParseResult &result) {
  // Print headers
  std::print("Headers: ");
  for (size_t i = 0; i < result.headers.size(); ++i) {
    std::cout << std::format("\"{}\"", result.headers[i]);
    if (i < result.headers.size() - 1) {
      std::print(", ");
    }
  }
  std::print("\n\n");

  // Print rows
  for (size_t row_idx = 0; row_idx < result.rows.size(); ++row_idx) {
    const auto &row = result.rows[row_idx];
    std::cout << std::format("Row {}: ", row_idx + 1);

    for (size_t col_idx = 0; col_idx < row.size(); ++col_idx) {
      std::cout << std::format("\"{}\"", row[col_idx]);
      if (col_idx < row.size() - 1) {
        std::print(", ");
      }
    }
    std::print("\n\n");
  }
}
