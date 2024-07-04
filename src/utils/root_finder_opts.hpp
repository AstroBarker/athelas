#ifndef ROOT_FINDER_OPTS_HPP_
#define ROOT_FINDER_OPTS_HPP_

/**
 * This file contains various solver options
 **/

#include "abstractions.hpp"

namespace Root_Finder_Opts {

constexpr unsigned int MAX_ITERS = 200;
constexpr Real FPTOL             = 1.0e-11;
constexpr Real ZBARTOL           = 1.0e-15;
constexpr Real ZBARTOLINV        = 1.0e15;

} // namespace Root_Finder_Opts

#endif // ROOT_FINDER_OPTS_HPP_
