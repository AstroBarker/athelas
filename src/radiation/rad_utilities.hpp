#ifndef RAD_UTILITIES_HPP_
#define RAD_UTILITIES_HPP_

#include <tuple>

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"

Real FluxFactor( const Real E, const Real F );
Real Flux_Rad( Real E, Real F, Real P, Real V, int iCR );
std::tuple<Real, Real> RadiationFourForce( const Real D, const Real V,
                                           const Real T, const Real kappa_r,
                                           const Real kappa_p, const Real E,
                                           const Real F, const Real Pr );
Real Source_Rad( const Real D, const Real V, const Real T, const Real kappa_r,
                 const Real kappa_p, const Real E, const Real F, const Real Pr,
                 const int iCR );
Real ComputeClosure( const Real E, const Real F );
Real Lambda_HLL( const Real f, const int sign );
void llf_flux( const Real Fp, const Real Fm, const Real Up, const Real Um,
               const Real alpha, Real &out );
void numerical_flux_hll_rad( const Real E_L, const Real E_R, const Real F_L,
                             const Real F_R, const Real P_L, const Real P_R,
                             Real &Flux_E, Real &Flux_F );
Real ComputeTimestep_Rad( const GridStructure *Grid, const Real CFL );
#endif // RAD_UTILITIES_HPP_
