#ifndef CHARACTERISTICDECOMPOSITION_H
#define CHARACTERISTICDECOMPOSITION_H

#include "Kokkos_Core.hpp"

void ComputeCharacteristicDecomposition( Kokkos::View<double[3]> U,
                                         Kokkos::View<double[3][3]> R,
                                         Kokkos::View<double[3][3]> R_inv );

#endif