/*
 * @file slope_limiter_unlimited.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief No-op slope limiter, used when limiting is disabled.
 * This does nothing.
 */

#include "slope_limiter.hpp"

/*
 * No op
 */
void Unlimited::apply_slope_limiter(View3D<double> /*U*/,
                                    const GridStructure* /*grid*/,
                                    const ModalBasis* /*basis*/,
                                    const EOS* /*eos*/) {}

auto Unlimited::get_limited(const int /*iX*/) const -> int { return 0.0; }
