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

namespace root_finders {

constexpr static unsigned int MAX_ITERS = 20;
constexpr static double ABSTOL          = 1.0e-13;
constexpr static double RELTOL          = 1.0e-13;
constexpr static double ZBARTOL         = 1.0e-15;
constexpr static double ZBARTOLINV      = 1.0e15;

} // namespace root_finders
