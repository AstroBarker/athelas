#pragma once
/**
 * @file quadrature.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Quadrature rules
 *
 * @details Computes Gauss-legendre nodes and weights
 */

#include "abstractions.hpp"
#include <vector>

namespace quadrature {
auto jacobi_matrix( int m, std::vector<double>& aj, std::vector<double>& bj )
    -> double;
void lg_quadrature( int m, std::vector<double>& nodes,
                    std::vector<double>& weights );
} // namespace quadrature
