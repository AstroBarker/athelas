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
auto jacobi_matrix( int m, std::vector<Real>& aj, std::vector<Real>& bj )
    -> Real;
void lg_quadrature( int m, std::vector<Real>& nodes,
                    std::vector<Real>& weights );
} // namespace quadrature
