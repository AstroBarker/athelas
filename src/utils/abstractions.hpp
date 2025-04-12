/**
 * @file abstractions.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Provides useful definitions.
 */

#ifndef ABSTRACTIONS_HPP_
#define ABSTRACTIONS_HPP_

#include "Kokkos_Core.hpp"

#include "Kokkos_Core.hpp"

using Real = double;

template <typename T>
using View4D = Kokkos::View<T ****>;

template <typename T>
using View3D = Kokkos::View<T ***>;

template <typename T>
using View2D = Kokkos::View<T **>;

template <typename T>
using View1D = Kokkos::View<T *>;

/* Where to put this? */
namespace PolyBasis {
enum PolyBasis { Legendre, Taylor };
}

#endif // ABSTRACTIONS_HPP_
