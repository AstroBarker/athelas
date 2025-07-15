#pragma once
/**
 * @file fluid_discretization.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Contains the main discretization routines for radiation.
 *
 * @details We implement the core DG updates for radiation here, including
 *          - ComputerIncrement_Rad_Divergence (hyperbolic term)
 *          - compute_increment_rad_source (coupling source term)
 */

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos.hpp"
#include "opacity/opac.hpp"
#include "opacity/opac_variant.hpp"

namespace radiation {

using bc::BoundaryConditions;

void compute_increment_rad_divergence(
    const View3D<double> uCR, const View3D<double> uCF, const GridStructure& grid,
    const ModalBasis* basis, const ModalBasis* fluid_basis, const EOS* eos, 
    View3D<double> dU, View3D<double> Flux_q, View2D<double> dFlux_num, 
    View2D<double> uCF_F_L, View2D<double> uCF_F_R, View1D<double> Flux_U, 
    View1D<double> Flux_P );

void compute_increment_rad_explicit(
    const View3D<double> uCR, const View3D<double> uCF, const GridStructure& grid,
    const ModalBasis* basis, const ModalBasis* fluid_basis, const EOS* eos, 
    View3D<double> dU, View3D<double> Flux_q, View2D<double> dFlux_num, 
    View2D<double> uCR_F_L, View2D<double> uCR_F_R, View1D<double> Flux_U, 
    View1D<double> Flux_P, const Options* opts, BoundaryConditions* bcs );

auto compute_increment_rad_source( View2D<double> uCR, int k, int iCR,
                                   View2D<double> uCF, const GridStructure& grid,
                                   const ModalBasis* fluid_basis, 
                                   const ModalBasis* rad_basis, const EOS* eos,
                                   const Opacity* opac, int iX ) -> double;
auto compute_increment_radhydro_source( View2D<double> uCRH, int k,
                                   const GridStructure& grid,
                                   const ModalBasis* fluid_basis, 
                                   const ModalBasis* rad_basis, const EOS* eos,
                                   const Opacity* opac, int iX ) -> std::tuple<double, double, double, double>;
} // namespace radiation
