#ifndef ROOT_FINDER_OPTS_HPP_
#define ROOT_FINDER_OPTS_HPP_

/**
 * This file contains various solver options
 **/

#include "abstractions.hpp"

namespace root_finders {

constexpr unsigned int MAX_ITERS = 200;
constexpr Real FPTOL             = 1.0e-10;
constexpr Real RELTOL            = 1.0e-8;
constexpr Real ZBARTOL           = 1.0e-15;
constexpr Real ZBARTOLINV        = 1.0e15;

} // namespace root_finders

#endif // ROOT_FINDER_OPTS_HPP_
