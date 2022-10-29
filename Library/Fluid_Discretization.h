#ifndef FLUID_DISCRETIZATION_H
#define FLUID_DISCRETIZATION_H

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"

void ComputeIncrement_Fluid_Divergence(
    const Kokkos::View<Real***> U, const GridStructure& Grid,
    const ModalBasis& Basis, Kokkos::View<Real***> dU,
    Kokkos::View<Real***> Flux_q, Kokkos::View<Real**> dFlux_num,
    Kokkos::View<Real**> uCF_F_L, Kokkos::View<Real**> uCF_F_R,
    Kokkos::View<Real*> Flux_U, Kokkos::View<Real*> Flux_P );

void ComputeIncrement_Fluid_Geometry( const Kokkos::View<Real***> U,
                                      const GridStructure& Grid,
                                      const ModalBasis& Basis,
                                      Kokkos::View<Real***> dU );

void Compute_Increment_Explicit(
    const Kokkos::View<Real***> U, const GridStructure& Grid,
    const ModalBasis& Basis, Kokkos::View<Real***> dU,
    Kokkos::View<Real***> Flux_q, Kokkos::View<Real**> dFlux_num,
    Kokkos::View<Real**> uCF_F_L, Kokkos::View<Real**> uCF_F_R,
    Kokkos::View<Real*> Flux_U, Kokkos::View<Real*> Flux_P,
    const std::string BC );

#endif
