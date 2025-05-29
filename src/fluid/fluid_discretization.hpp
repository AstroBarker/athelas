#pragma once
/**
 * @file fluid_discretization.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Contains the main discretization routines for the fluid
 *
 * @details We implement the core DG updates for the fluid here, including
 *            - ComputerIncrement_Fluid_Divergence (hyperbolic term)
 *            - compute_increment_fluid_geometry (geometric source)
 *            - compute_increment_fluid_rad (radiation source term)
 */

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos.hpp"
#include "opacity/opac_variant.hpp"

namespace fluid {

using bc::BoundaryConditions;

void compute_increment_fluid_divergence(
    View3D<double> U, const GridStructure& grid, const ModalBasis* Basis,
    const EOS* eos, View3D<double> dU, View3D<double> Flux_q,
    View2D<double> dFlux_num, View2D<double> uCF_F_L, View2D<double> uCF_F_R,
    View1D<double> Flux_U, View1D<double> Flux_P );

void compute_increment_fluid_geometry( View3D<double> U,
                                       const GridStructure& grid,
                                       ModalBasis* Basis, EOS* eos,
                                       View3D<double> dU );

auto compute_increment_fluid_source( View2D<double> uCF, int k, int iCF,
                                     View2D<double> uCR,
                                     const GridStructure& grid,
                                     const ModalBasis* fluid_basis, 
                                     const ModalBasis* rad_basis, const EOS* eos,
                                     const Opacity* opac, int iX ) -> double;

void compute_increment_fluid_explicit(
    View3D<double> U, const GridStructure& grid,
    const ModalBasis* Basis, const EOS* eos, View3D<double> dU,
    View3D<double> Flux_q, View2D<double> dFlux_num, View2D<double> uCF_F_L,
    View2D<double> uCF_F_R, View1D<double> Flux_U, View1D<double> Flux_P,
    const Options* opts, BoundaryConditions* bcs );

} // namespace fluid
