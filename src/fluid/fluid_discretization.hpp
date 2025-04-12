#ifndef FLUID_DISCRETIZATION_HPP_
#define FLUID_DISCRETIZATION_HPP_
/**
 * @file fluid_discretization.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Contains the main discretization routines for the fluid
 * 
 * @details We implement the core DG updates for the fluid here, including
 *            - ComputerIncrement_Fluid_Divergence (hyperbolic term)
 *            - ComputeIncrement_Fluid_Geometry (geometric source)
 *            - ComputeIncrement_Fluid_Rad (radiation source term)
 */

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "eos.hpp"
#include "opacity/opac_variant.hpp"

void ComputeIncrement_Fluid_Divergence(
    const View3D<Real> U, GridStructure &Grid, const ModalBasis *Basis,
    const EOS *eos, View3D<Real> dU, View3D<Real> Flux_q,
    View2D<Real> dFlux_num, View2D<Real> uCF_F_L, View2D<Real> uCF_F_R,
    View1D<Real> Flux_U, View1D<Real> Flux_P, const Options opts );

void ComputeIncrement_Fluid_Geometry( const View3D<Real> U, GridStructure &Grid,
                                      ModalBasis *Basis, EOS *eos,
                                      View3D<Real> dU );

Real ComputeIncrement_Fluid_Rad( View2D<Real> uCF, const int k, const int iCF,
                                 const View2D<Real> uCR, GridStructure &Grid,
                                 const ModalBasis *Basis, const EOS *eos,
                                 const Opacity *opac, const int iX );

void Compute_Increment_Explicit( const View3D<Real> U, const View3D<Real> uCR,
                                 GridStructure &Grid, const ModalBasis *Basis,
                                 const EOS *eos, View3D<Real> dU,
                                 View3D<Real> Flux_q, View2D<Real> dFlux_num,
                                 View2D<Real> uCF_F_L, View2D<Real> uCF_F_R,
                                 View1D<Real> Flux_U, View1D<Real> Flux_P,
                                 const Options opts );

#endif // FLUID_DISCRETIZATION_HPP_
