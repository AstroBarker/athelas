#ifndef ABSTRACTIONS_H
#define ABSTRACTIONS_H

#include "Kokkos_Core.hpp"

using Real = double;
using UInt = unsigned int;
using View3D = Kokkos::View<Real ***>;

/* Where to put this? */
namespace PolyBasis {
  enum PolyBasis { Legendre, Taylor };
}

#endif
