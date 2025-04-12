#ifndef SLOPE_LIMITER_BASE_HPP_
#define SLOPE_LIMITER_BASE_HPP_
/**
 * @file slope_limiter_base.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Base class for slope limiters.
 * 
 * @details Defines the SlopeLimiterBase template class that serves 
 *          as the foundation for all slope limiters implemented.
 * 
 *          The class provides two key interface methods:
 *          - ApplySlopeLimiter: Applies the limiter to the solution
 *          - Get_Limited: Returns whether a cell was limited
 */

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
