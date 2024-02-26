#ifndef _RAD_DISCRETIZATION_HPP_
#define _RAD_DISCRETIZATION_HPP_

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "EoS.hpp"

void ComputeIncrement_Rad_Divergence( const View3D uCR, const View3D uCF,
                                      GridStructure &rid, ModalBasis *Basis,
                                      EOS *eos, View3D dU, View3D Flux_q,
                                      View2D dFlux_num, View2D uCF_F_L,
                                      View2D uCF_F_R, View1D Flux_U,
                                      View1D Flux_P );

void Compute_Increment_Explicit_Rad( const View3D uCR, const View3D uCF,
                                     GridStructure &Grid, ModalBasis *Basis,
                                     EOS *eos, View3D dU, View3D Flux_q,
                                     View2D dFlux_num, View2D uCR_F_L,
                                     View2D uCR_F_R, View1D Flux_U,
                                     View1D Flux_P, const Options opts );

void ComputeIncrement_Rad_Source( View3D uCR, View3D uCF, GridStructure &Grid,
                                  ModalBasis *Basis, EOS *eos, View3D dU );
#endif // _RAD_DISCRETIZATION_HPP_
