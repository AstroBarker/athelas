#ifndef RAD_DISCRETIZATION_HPP_
#define RAD_DISCRETIZATION_HPP_
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
#include "eos.hpp"
#include "opacity/opac.hpp"
#include "opacity/opac_variant.hpp"

namespace radiation {

void compute_increment_rad_divergence(
    View3D<Real> uCR, View3D<Real> uCF, const GridStructure& rid,
    const ModalBasis* Basis, const EOS* eos, View3D<Real> dU,
    View3D<Real> Flux_q, View2D<Real> dFlux_num, View2D<Real> uCF_F_L,
    View2D<Real> uCF_F_R, View1D<Real> Flux_U, View1D<Real> Flux_P );

void compute_increment_explicit_rad( View3D<Real> uCR, View3D<Real> uCF,
                                     const GridStructure& Grid,
                                     const ModalBasis* Basis, const EOS* eos,
                                     View3D<Real> dU, View3D<Real> Flux_q,
                                     View2D<Real> dFlux_num,
                                     View2D<Real> uCR_F_L, View2D<Real> uCR_F_R,
                                     View1D<Real> Flux_U, View1D<Real> Flux_P,
                                     const Options* opts );

auto compute_increment_rad_source( View2D<Real> uCR, int k, int iCR,
                                   View2D<Real> uCF, const GridStructure& Grid,
                                   const ModalBasis* Basis, const EOS* eos,
                                   const Opacity* opac, int iX ) -> Real;
} // namespace radiation
#endif // RAD_DISCRETIZATION_HPP_
