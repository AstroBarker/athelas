/**
 * @file fluid_utilities.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Utilities for fluid evolution
 *
 * @details Contains functions necessary for fluid evolution
 */

#include <algorithm> // std::min, std::max
#include <cstdlib> /* abs */
#include <iostream>
#include <vector>

#include "constants.hpp"
#include "eos.hpp"
#include "error.hpp"
#include "fluid_utilities.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"
#include "rad_utilities.hpp"
#include "utils/utilities.hpp"

using utilities::pos_part;

namespace fluid {
/**
 * Return a component iCF of the flux vector.
 * TODO: flux_fluid needs streamlining
 **/
auto flux_fluid( const Real V, const Real P, const int iCF ) -> Real {
  assert( iCF == 0 || iCF == 1 || iCF == 2 );
  assert( P > 0.0 && "Flux_Flux :: negative pressure" );
  if ( iCF == 0 ) {
    return -V;
  }
  if ( iCF == 1 ) {
    return +P;
  } else if ( iCF == 2 ) {
    return +P * V;
  } else { // Error case. Shouldn't ever trigger.
    THROW_ATHELAS_ERROR( " ! Please input a valid iCF! (0,1,2). " );
    return -1.0; // just a formality.
  }
}

/**
 * Fluid radiation sources. Kind of redundant with Rad_sources.
 **/
auto source_fluid_rad( const Real D, const Real V, const Real T,
                       const Real kappa_r, const Real kappa_p, const Real E,
                       const Real F, const Real Pr, const int iCF ) -> Real {
  assert( iCF == 1 || iCF == 2 );

  constexpr static Real c = constants::c_cgs;

  auto [G0, G] =
      radiation::radiation_four_force( D, V, T, kappa_r, kappa_p, E, F, Pr );

  return ( iCF == 1 ) ? G : c * G0;
}

/**
 * Positivity preserving numerical flux. Constructs v* and p* states.
 * TODO(astrobarker): do I need tau_r_star if I construct p* with left?
 **/
auto numerical_flux_gudonov_positivity( const Real tauL, const Real tauR, 
                             const Real vL, const Real vR, const Real pL,
                             const Real pR, const Real csL, 
                             const Real csR ) -> std::tuple<Real, Real> {
  assert( pL > 0.0 && pR > 0.0 &&
          "numerical_flux_gudonov :: negative pressure" );
  const Real pRmL = pR - pL; // [[p]]
  const Real vRmL = vR - vL; // [[v]]
  const Real zL = std::max(std::max(std::sqrt(pos_part(pRmL) / tauL), -(vRmL)/tauL), csL/tauL);
  const Real zR = std::max(std::max(std::sqrt(pos_part(-pR + pL) / tauR), -(vRmL)/tauR), csR/tauR);
  const Real z_sum = zL + zR;

  // get tau star states
  const Real term1_l = tauL - (pRmL) / (zL * zL);
  const Real term2_l = tauL + vRmL / zL;
  const Real tau_l_star = (zL * term1_l + zR * term2_l) / z_sum; 

  /*
  const Real term1_r = tauR + vRmL / zR;
  const Real term2_r = tauR + pRmL / (zR * zR);
  const Real tau_r_star = (zL * term1_r + zR * term2_r) / z_sum;
  */

  // vstar, pstar
  const Real Flux_U = ( -pRmL + zR * vR + zL * vL ) / ( z_sum );
  const Real Flux_P = pL - (zL * zL) * (tau_l_star - tauL);
  return {Flux_U, Flux_P};
}

/**
 * Gudonov style numerical flux. Constructs v* and p* states.
 **/
auto numerical_flux_gudonov( const Real vL, const Real vR, const Real pL,
                             const Real pR, const Real zL, const Real zR ) -> std::tuple<Real, Real> {
  assert( pL > 0.0 && pR > 0.0 &&
          "numerical_flux_gudonov :: negative pressure" );
  const Real Flux_U = ( pL - pR + zR * vR + zL * vL ) / ( zR + zL );
  const Real Flux_P = ( zR * pL + zL * pR + zL * zR * ( vL - vR ) ) / ( zR + zL );
  return {Flux_U, Flux_P};
}

/**
 * Gudonov style numerical flux. Constructs v* and p* states.
 **/
void numerical_flux_hllc( Real vL, Real vR, Real pL, Real pR, Real cL, Real cR,
                          Real rhoL, Real rhoR, Real& Flux_U, Real& Flux_P ) {
  Real const aL = vL - cL; // left wave speed estimate
  Real const aR = vR + cR; // right wave speed estimate
  Flux_U = ( rhoR * vR * ( aR - vR ) - rhoL * vL * ( aL - vL ) + pL - pR ) /
           ( rhoR * ( aR - vR ) - rhoL * ( aL - vL ) );
  Flux_P = rhoL * ( vL - aL ) * ( vL - Flux_U ) + pL;
}

// Compute Auxiliary

/**
 * Compute the fluid timestep.
 **/
auto compute_timestep_fluid( const View3D<Real> U, const GridStructure* grid,
                             EOS* eos, const Real CFL ) -> Real {

  const Real MIN_DT = 1.0e-14;
  const Real MAX_DT = 100.0;

  const int& ilo = grid->get_ilo( );
  const int& ihi = grid->get_ihi( );

  Real dt = 0.0;
  Kokkos::parallel_reduce(
      "Compute Timestep", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
      KOKKOS_LAMBDA( const int iX, Real& lmin ) {
        // --- Compute Cell Averages ---
        Real tau_x  = U( 0, iX, 0 );
        Real vel_x  = U( 1, iX, 0 );
        Real eint_x = U( 2, iX, 0 );

        assert( tau_x > 0.0 && "Compute Timestep :: bad specific volume" );
        assert( eint_x > 0.0 && "Compute Timestep :: bad specific energy" );

        Real dr = grid->get_widths( iX );

        auto lambda = nullptr;
        const Real Cs =
            eos->sound_speed_from_conserved( tau_x, vel_x, eint_x, lambda );
        Real eigval = Cs + std::abs( vel_x );

        Real dt_old = std::abs( dr ) / std::abs( eigval );

        if ( dt_old < lmin ) lmin = dt_old;
      },
      Kokkos::Min<Real>( dt ) );

  dt = std::max( CFL * dt, MIN_DT );
  dt = std::min( dt, MAX_DT );

  assert( !std::isnan( dt ) && "NaN encountered in compute_timestep_fluid.\n" );

  return dt;
}
} // namespace fluid
