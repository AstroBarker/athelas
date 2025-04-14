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
 *            - compute_increment_fluid_geometry (geometric source)
 *            - compute_increment_fluid_rad (radiation source term)
 */

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "eos.hpp"
#include "opacity/opac_variant.hpp"

namespace fluid {

void compute_increment_fluid_divergence(
    View3D<Real> U, const GridStructure& grid, const ModalBasis* Basis,
    const EOS* eos, View3D<Real> dU, View3D<Real> Flux_q,
    View2D<Real> dFlux_num, View2D<Real> uCF_F_L, View2D<Real> uCF_F_R,
    View1D<Real> Flux_U, View1D<Real> Flux_P );

void compute_increment_fluid_geometry( View3D<Real> U,
                                       const GridStructure& grid,
                                       ModalBasis* Basis, EOS* eos,
                                       View3D<Real> dU );

auto compute_increment_fluid_rad( View2D<Real> uCF, int k, int iCF,
                                  View2D<Real> uCR, const GridStructure& grid,
                                  const ModalBasis* Basis, const EOS* eos,
                                  const Opacity* opac, int iX ) -> Real;

void compute_increment_explicit( View3D<Real> U, View3D<Real> uCR,
                                 const GridStructure& grid,
                                 const ModalBasis* Basis, const EOS* eos,
                                 View3D<Real> dU, View3D<Real> Flux_q,
                                 View2D<Real> dFlux_num, View2D<Real> uCF_F_L,
                                 View2D<Real> uCF_F_R, View1D<Real> Flux_U,
                                 View1D<Real> Flux_P, const Options* opts );

} // namespace fluid
#endif // FLUID_DISCRETIZATION_HPP_
