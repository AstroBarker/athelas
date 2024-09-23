#ifndef RAD_DISCRETIZATION_HPP_
#define RAD_DISCRETIZATION_HPP_

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "eos.hpp"

void ComputeIncrement_Rad_Divergence(
    const View3D<Real> uCR, const View3D<Real> uCF, GridStructure &rid,
    const ModalBasis *Basis, const EOS *eos, View3D<Real> dU,
    View3D<Real> Flux_q, View2D<Real> dFlux_num, View2D<Real> uCF_F_L,
    View2D<Real> uCF_F_R, View1D<Real> Flux_U, View1D<Real> Flux_P );

void Compute_Increment_Explicit_Rad(
    const View3D<Real> uCR, const View3D<Real> uCF, GridStructure &Grid,
    const ModalBasis *Basis, const EOS *eos, View3D<Real> dU,
    View3D<Real> Flux_q, View2D<Real> dFlux_num, View2D<Real> uCR_F_L,
    View2D<Real> uCR_F_R, View1D<Real> Flux_U, View1D<Real> Flux_P,
    const Options opts );

Real ComputeIncrement_Rad_Source( View2D<Real> uCR, const int k, const int iCR,
                                  const View2D<Real> uCF, GridStructure &Grid,
                                  const ModalBasis *Basis, const EOS *eos,
                                  const int iX );
#endif // RAD_DISCRETIZATION_HPP_
