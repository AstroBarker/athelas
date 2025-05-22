#pragma once
/**
 * @file root_finder_opts.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Root finder options
 *
 * @details Compile time root finder options
 *          - FPTOL (absolute tolerance)
 *          - RELTOL (relative tolerance)
 *
 *          TODO: these should be runtime..
 */

#include "abstractions.hpp"

namespace root_finders {

constexpr static unsigned int MAX_ITERS = 200;
constexpr static Real FPTOL             = 1.0e-10;
constexpr static Real RELTOL            = 1.0e-14;
constexpr static Real ZBARTOL           = 1.0e-15;
constexpr static Real ZBARTOLINV        = 1.0e15;

} // namespace root_finders
