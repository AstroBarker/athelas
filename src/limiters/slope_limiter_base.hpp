#ifndef SLOPE_LIMITER_BASE_HPP_
#define SLOPE_LIMITER_BASE_HPP_

/**
 * define a base class using curiously recurring template pattern
 **/
#include "abstractions.hpp"
#include "polynomial_basis.hpp"

template <class SlopeLimiter>
class SlopeLimiterBase {
 public:
  void ApplySlopeLimiter( View3D<Real> U, const GridStructure *grid,
                          const ModalBasis *basis ) const {
    return static_cast<SlopeLimiter const *>( this )->ApplySlopeLimiter(
        U, grid, basis );
  }
  int Get_Limited( const int iX ) const {
    return static_cast<SlopeLimiter const *>( this )->Get_Limited( iX );
  }
};

#endif // SLOPE_LIMITER_BASE_HPP_
