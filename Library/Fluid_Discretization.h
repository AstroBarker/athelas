#ifndef FLUID_DISCRETIZATION_H
#define FLUID_DISCRETIZATION_H

#include "Kokkos_Core.hpp"

void ComputeIncrement_Fluid_Divergence(
    Kokkos::View<double***> U, GridStructure& Grid, ModalBasis& Basis,
    Kokkos::View<double***> dU, Kokkos::View<double***> Flux_q,
    Kokkos::View<double**> dFlux_num, Kokkos::View<double**> uCF_F_L,
    Kokkos::View<double**> uCF_F_R, Kokkos::View<double*> Flux_U,
    Kokkos::View<double*> Flux_P );

void ComputeIncrement_Fluid_Geometry( Kokkos::View<double***> U,
                                      GridStructure& Grid, ModalBasis& Basis,
                                      Kokkos::View<double***> dU );

void Compute_Increment_Explicit(
    Kokkos::View<double***> U, GridStructure& Grid, ModalBasis& Basis,
    Kokkos::View<double***> dU, Kokkos::View<double***> Flux_q,
    Kokkos::View<double**> dFlux_num, Kokkos::View<double**> uCF_F_L,
    Kokkos::View<double**> uCF_F_R, Kokkos::View<double*> Flux_U,
    Kokkos::View<double*> Flux_P, const std::string BC );

#endif