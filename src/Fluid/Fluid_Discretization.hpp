#ifndef _FLUID_DISCRETIZATION_HPP_
#define _FLUID_DISCRETIZATION_HPP_

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "EoS.hpp"

void ComputeIncrement_Fluid_Divergence(
    const View3D<Real> U, GridStructure &Grid, const ModalBasis *Basis,
    const EOS *eos, View3D<Real> dU, View3D<Real> Flux_q,
    View2D<Real> dFlux_num, View2D<Real> uCF_F_L, View2D<Real> uCF_F_R,
    View1D<Real> Flux_U, View1D<Real> Flux_P, const Options opts );

void ComputeIncrement_Fluid_Geometry( const View3D<Real> U, GridStructure &Grid,
                                      ModalBasis *Basis, EOS *eos,
                                      View3D<Real> dU );

// void Compute_Increment_Explicit(
//     const Kokkos::View<Real ***> U, GridStructure &Grid, ModalBasis *Basis,
//     EOS *eos, Kokkos::View<Real ***> dU, Kokkos::View<Real ***> Flux_q,
//     Kokkos::View<Real **> dFlux_num, Kokkos::View<Real **> uCF_F_L,
//     Kokkos::View<Real **> uCF_F_R, Kokkos::View<Real *> Flux_U,
//     Kokkos::View<Real *> Flux_P, const std::string BC );

void ComputeIncrement_Fluid_Rad( const View3D<Real> uCF, const View3D<Real> uCR,
                                 GridStructure &Grid, const ModalBasis *Basis,
                                 View3D<Real> dU );

void Compute_Increment_Explicit( const View3D<Real> U, const View3D<Real> uCR,
                                 GridStructure &Grid, const ModalBasis *Basis,
                                 const EOS *eos, View3D<Real> dU,
                                 View3D<Real> Flux_q, View2D<Real> dFlux_num,
                                 View2D<Real> uCF_F_L, View2D<Real> uCF_F_R,
                                 View1D<Real> Flux_U, View1D<Real> Flux_P,
                                 const Options opts );

#endif // _FLUID__DISCRETIZATION_HPP_
