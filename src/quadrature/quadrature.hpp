#ifndef _QUADRATURE_HPP_
#define _QUADRATURE_HPP_

#include "abstractions.hpp"

namespace quadrature {
Real Jacobi_Matrix( int m, Real *aj, Real *bj );
void LG_Quadrature( int m, Real *nodes, Real *weights );
} // namespace quadrature

#endif // _QUADRATURE_HPP_
