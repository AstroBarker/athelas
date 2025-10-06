/**
 * @file quadrature.cpp
 * --------------
 *
 * @brief Quadrature rules
 *
 * @details Computes Gauss-legendre nodes and weights
 */

#pragma once

#include <vector>

namespace athelas::quadrature {
auto jacobi_matrix(int m, std::vector<double> &aj, std::vector<double> &bj)
    -> double;
void lg_quadrature(int m, std::vector<double> &nodes,
                   std::vector<double> &weights);
} // namespace athelas::quadrature
