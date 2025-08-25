#include "limiters/slope_limiter.hpp"

/*
 * No op
 */
void Unlimited::apply_slope_limiter(View3D<double> /*U*/,
                                    const GridStructure* /*grid*/,
                                    const ModalBasis* /*basis*/,
                                    const EOS* /*eos*/) {}

auto Unlimited::get_limited(const int /*iX*/) const -> int { return 0.0; }
