#ifndef QUADRATURE_HPP_
#define QUADRATURE_HPP_
/**
 * @file quadrature.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Quadrature rules
 *
 * @details Computes Gauss-Legendre nodes and weights
 */

#include "abstractions.hpp"

namespace quadrature {
Real Jacobi_Matrix( int m, Real *aj, Real *bj );
void LG_Quadrature( int m, Real *nodes, Real *weights );
} // namespace quadrature

#endif // QUADRATURE_HPP_
