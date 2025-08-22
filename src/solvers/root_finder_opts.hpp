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

static constexpr unsigned int MAX_ITERS = 200;
static constexpr double ABSTOL = 1.0e-10;
static constexpr double RELTOL = 1.0e-10;
static constexpr double ZBARTOL = 1.0e-15;
static constexpr double ZBARTOLINV = 1.0e15;

} // namespace root_finders
