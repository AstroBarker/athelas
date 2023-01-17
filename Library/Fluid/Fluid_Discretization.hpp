#ifndef _FLUID_DISCRETIZATION_HPP_
#define _FLUID_DISCRETIZATION_HPP_

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"

void ComputeIncrement_Fluid_Divergence(
    const View3D U, GridStructure *Grid, ModalBasis *Basis,
    View3D dU, View3D Flux_q, View2D dFlux_num, View2D uCF_F_L,
    View2D uCF_F_R, View1D Flux_U,
    View1D Flux_P, const Options opts );

void ComputeIncrement_Fluid_Geometry( const View3D U,
                                      GridStructure *Grid, ModalBasis *Basis,
                                      View3D dU );

void ComputeIncrement_Fluid_Rad( const View3D uCF, const View3D uCR,
                                 GridStructure *Grid, ModalBasis *Basis,
                                 View3D dU );

void Compute_Increment_Explicit(
    const View3D U, const View3D uCR, GridStructure *Grid, ModalBasis *Basis,
    View3D dU, View3D Flux_q,
    View2D dFlux_num, View2D uCF_F_L,
    View2D uCF_F_R, View1D Flux_U,
    View1D Flux_P, const Options opts );

#endif // _FLUID__DISCRETIZATION_HPP_
