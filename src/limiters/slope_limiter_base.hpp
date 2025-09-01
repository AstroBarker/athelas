#pragma once
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
 *          - apply_slope_limiter: Applies the limiter to the solution
 *          - get_limited: Returns whether a cell was limited
 */

#include "abstractions.hpp"
#include "eos/eos_variant.hpp"
#include "polynomial_basis.hpp"

template <class SlopeLimiter>
class SlopeLimiterBase {
 public:
  void apply_slope_limiter(View3D<double> U, const GridStructure* grid,
                           const ModalBasis* basis, const EOS* eos,
                           const std::vector<int>& vars) const {
    return static_cast<SlopeLimiter const*>(this)->apply_slope_limiter(
        U, grid, basis, eos, vars);
  }
  [[nodiscard]] auto get_limited(const int ix) const -> int {
    return static_cast<SlopeLimiter const*>(this)->get_limited(ix);
  }
  [[nodiscard]] auto limited() const -> View1D<int> {
    return static_cast<SlopeLimiter const*>(this)->limited();
  }
};
