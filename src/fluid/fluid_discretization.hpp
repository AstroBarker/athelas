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

namespace fluid {

void ComputeIncrement_Fluid_Divergence(
    View3D<Real> U, GridStructure& Grid, const ModalBasis* Basis,
    const EOS* eos, View3D<Real> dU, View3D<Real> Flux_q,
    View2D<Real> dFlux_num, View2D<Real> uCF_F_L, View2D<Real> uCF_F_R,
    View1D<Real> Flux_U, View1D<Real> Flux_P );

void ComputeIncrement_Fluid_Geometry( View3D<Real> U, GridStructure& Grid,
                                      ModalBasis* Basis, EOS* eos,
                                      View3D<Real> dU );

auto ComputeIncrement_Fluid_Rad( View2D<Real> uCF, int k, int iCF,
                                 View2D<Real> uCR, GridStructure& Grid,
                                 const ModalBasis* Basis, const EOS* eos,
                                 const Opacity* opac, int iX ) -> Real;

void Compute_Increment_Explicit( View3D<Real> U, View3D<Real> uCR,
                                 GridStructure& Grid, const ModalBasis* Basis,
                                 const EOS* eos, View3D<Real> dU,
                                 View3D<Real> Flux_q, View2D<Real> dFlux_num,
                                 View2D<Real> uCF_F_L, View2D<Real> uCF_F_R,
                                 View1D<Real> Flux_U, View1D<Real> Flux_P,
                                 const Options* opts );

} // namespace fluid
#endif // FLUID_DISCRETIZATION_HPP_
