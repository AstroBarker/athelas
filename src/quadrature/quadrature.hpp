#ifndef QUADRATURE_HPP_
#define QUADRATURE_HPP_
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

namespace quadrature {
auto jacobi_matrix( int m, Real* aj, Real* bj ) -> Real;
void lg_quadrature( int m, Real* nodes, Real* weights );
} // namespace quadrature

#endif // QUADRATURE_HPP_
