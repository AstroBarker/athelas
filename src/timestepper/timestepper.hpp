#ifndef TIMESTEPPER_HPP_
#define TIMESTEPPER_HPP_
/**
 * @file timestepper.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Primary time marching routine.
 *
 * @details Timestppers for hydro and rad hydro.
 *          Uses explicity for transport terms and implicit for coupling.
 */

#include <math.h>

#include "abstractions.hpp"
#include "bound_enforcing_limiter.hpp"
#include "eos.hpp"
#include "opacity/opac.hpp"
#include "opacity/opac_variant.hpp"
#include "polynomial_basis.hpp"
#include "problem_in.hpp"
#include "slope_limiter.hpp"
#include "slope_limiter_base.hpp"
#include "solvers/root_finders.hpp"
#include "state.hpp"
#include "tableau.hpp"

class TimeStepper {

 public:
  // TODO(astrobarker): Is it possible to initialize Grid_s from Grid directly?
  TimeStepper( ProblemIn* pin, GridStructure& Grid );

  void initialize_timestepper( );

  /**
   * Update fluid solution with SSPRK methods
   **/
  template <typename T>
  void update_fluid( T ComputeIncrement, const Real dt, State* state,
                     GridStructure& Grid, const ModalBasis* Basis,
                     const EOS* eos, SlopeLimiter* S_Limiter,
                     const Options* opts ) {

    // hydro explicity update
    update_fluid_explicit( ComputeIncrement, dt, state, Grid, Basis, eos,
                           S_Limiter, opts );
  }

  /**
   * Explicit fluid update with SSPRK methods
   **/
  template <typename T>
  void update_fluid_explicit( T ComputeIncrement, const Real dt, State* state,
                              GridStructure& Grid, const ModalBasis* Basis,
                              const EOS* eos, SlopeLimiter* S_Limiter,
                              const Options* opts ) {

    const auto& order = Basis->get_order( );
    const auto& ihi   = Grid.get_ihi( );

    auto U   = state->get_u_cf( );
    auto uCR = state->get_u_cr( );

    const int nvars = U.extent( 0 );

    Grid_s[0] = Grid;

    for ( unsigned short int iS = 0; iS < nStages; iS++ ) {
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { nvars, ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iCF, const int iX, const int k ) {
            SumVar_U( iCF, iX, k ) = U( iCF, iX, k );
            StageData( iS, iX )    = Grid.get_left_interface( iX );
            // StageData( iS, iX )    = Grid_s[iS].get_left_interface( iX );
          } );

      // --- Inner update loop ---

      for ( int j = 0; j < iS; j++ ) {
        auto Us_j =
            Kokkos::subview( U_s, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto dUs_j =
            Kokkos::subview( dU_s, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto Flux_Uj = Kokkos::subview( Flux_U, j, Kokkos::ALL );
        ComputeIncrement( Us_j, uCR, Grid_s[j], Basis, eos, dUs_j, Flux_q,
                          dFlux_num, uCF_F_L, uCF_F_R, Flux_Uj, Flux_P, opts );

        // inner sum
        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                    { nvars, ihi + 2, order } ),
            KOKKOS_CLASS_LAMBDA( const int iCF, const int iX, const int k ) {
              SumVar_U( iCF, iX, k ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * dUs_j( iCF, iX, k );
            } );

        Kokkos::parallel_for(
            "Timestepper::StageData", ihi + 2,
            KOKKOS_CLASS_LAMBDA( const int iX ) {
              StageData( iS, iX ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * Flux_Uj( iX );
            } );
      } // End inner loop

      // set U_s
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { nvars, ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iCF, const int iX, const int k ) {
            U_s( iS, iCF, iX, k ) = SumVar_U( iCF, iX, k );
          } );

      auto StageDataj = Kokkos::subview( StageData, iS, Kokkos::ALL );
      Grid_s[iS].update_grid( StageDataj );

      auto Us_j =
          Kokkos::subview( U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      // S_Limiter->apply_slope_limiter( Us_j, &Grid_s[iS], Basis );
      apply_slope_limiter( S_Limiter, Us_j, &Grid_s[iS], Basis );
      bel::apply_bound_enforcing_limiter( Us_j, Basis, eos );
    } // end outer loop

    for ( unsigned short int iS = 0; iS < nStages; iS++ ) {
      auto Us_j =
          Kokkos::subview( U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto dUs_j =
          Kokkos::subview( dU_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto Flux_Uj = Kokkos::subview( Flux_U, iS, Kokkos::ALL );

      ComputeIncrement( Us_j, uCR, Grid_s[iS], Basis, eos, dUs_j, Flux_q,
                        dFlux_num, uCF_F_L, uCF_F_R, Flux_Uj, Flux_P, opts );
      Kokkos::parallel_for(
          "Timestepper :: u^(n+1) from the stages",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { nvars, ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iCF, const int iX, const int k ) {
            U( iCF, iX, k ) +=
                dt * explicit_tableau_.b_i( iS ) * dUs_j( iCF, iX, k );
          } );

      Kokkos::parallel_for(
          "Timestepper::StageData::final", ihi + 2,
          KOKKOS_CLASS_LAMBDA( const int iX ) {
            StageData( 0, iX ) +=
                dt * Flux_Uj( iX ) * explicit_tableau_.b_i( iS );
          } );
      auto StageDataj = Kokkos::subview( StageData, 0, Kokkos::ALL );
      Grid_s[iS].update_grid( StageDataj );
    }

    Grid = Grid_s[nStages - 1];
    apply_slope_limiter( S_Limiter, U, &Grid, Basis );
    bel::apply_bound_enforcing_limiter( U, Basis, eos );
  }

  /**
   * Update radiation solution with SSPRK methods
   **/
  template <typename T>
  void update_radiation( T ComputeIncrementRad, const Real dt, State* state,
                         GridStructure& Grid, const ModalBasis* Basis,
                         const EOS* eos, SlopeLimiter* S_Limiter,
                         const Options opts ) {
    update_radiation_imex( ComputeIncrementRad, dt, state, Grid, Basis, eos,
                           S_Limiter, opts );
  }
  /**
   * Update radiation solution with SSPRK methods
   **/
  template <typename T>
  void update_radiation_imex( T ComputeIncrementRad, const Real dt,
                              State* state, GridStructure& Grid,
                              const ModalBasis* Basis, const EOS* eos,
                              SlopeLimiter* /*S_Limiter*/,
                              const Options& opts ) {

    const auto& order = Basis->get_order( );
    const auto& ihi   = Grid.get_ihi( );

    auto uCF = state->get_u_cf( );
    auto uCR = state->get_u_cr( );

    const int nvars = uCR.extent( 0 );

    for ( unsigned short int iS = 0; iS < nStages; iS++ ) {
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper :: Rad :: 1",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { nvars, ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iCR, const int iX, const int k ) {
            SumVar_U( iCR, iX, k ) = uCR( iCR, iX, k );
          } );

      // --- Inner update loop ---

      for ( int j = 0; j < iS; j++ ) {
        auto Us_j =
            Kokkos::subview( U_s_r, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto dUs_j =
            Kokkos::subview( dU_s_r, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto Flux_Uj = Kokkos::subview( Flux_U, j, Kokkos::ALL );
        ComputeIncrementRad( Us_j, uCF, Grid, Basis, eos, dUs_j, Flux_q,
                             dFlux_num, uCF_F_L, uCF_F_R, Flux_Uj, Flux_P,
                             opts );

        // inner sum
        Kokkos::parallel_for(
            "Timestepper :: Rad :: 2",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                    { nvars, ihi + 2, order } ),
            KOKKOS_CLASS_LAMBDA( const int iCR, const int iX, const int k ) {
              SumVar_U( iCR, iX, k ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * dUs_j( iCR, iX, k );
            } );
      } // End inner loop

      // set U_s
      Kokkos::parallel_for(
          "Timestepper :: Rad :: 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { nvars, ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iCR, const int iX, const int k ) {
            U_s_r( iS, iCR, iX, k ) = SumVar_U( iCR, iX, k );
          } );

      auto Us_j =
          Kokkos::subview( U_s_r, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      // S_Limiter->apply_slope_limiter( Us_j, &Grid, Basis );
      // apply_bound_enforcing_limiter( Us_j, Basis, eos );
    } // end outer loop

    for ( unsigned short int iS = 0; iS < nStages; iS++ ) {
      auto Us_j =
          Kokkos::subview( U_s_r, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto dUs_j =
          Kokkos::subview( dU_s_r, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto Flux_Uj = Kokkos::subview( Flux_U, iS, Kokkos::ALL );

      ComputeIncrementRad( Us_j, uCF, Grid, Basis, eos, dUs_j, Flux_q,
                           dFlux_num, uCF_F_L, uCF_F_R, Flux_Uj, Flux_P, opts );
      Kokkos::parallel_for(
          "Timestepper :: Rad :: u^(n+1) from the stages",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { nvars, ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iCR, const int iX, const int k ) {
            const Real dt_b = dt * explicit_tableau_.b_i( iS );
            uCR( iCR, iX, k ) += dt_b * dUs_j( iCR, iX, k );
          } );
    }

    // S_Limiter->apply_slope_limiter( uCR, &Grid, Basis );
    // apply_bound_enforcing_limiter( uCR, Basis, eos );
  }

  /**
   * Update fluid solution with SSPRK methods
   * TODO: register increment funcs with class?
   **/
  template <typename T, typename F, typename G, typename H>
  void update_rad_hydro( T compute_increment_hydro_explicit,
                         F compute_increment_rad_explicit,
                         G compute_increment_hydro_implicit,
                         H compute_increment_rad_implicit, const Real dt,
                         State* state, GridStructure& Grid,
                         const ModalBasis* Basis, const EOS* eos,
                         const Opacity* opac, SlopeLimiter* S_Limiter,
                         const Options* opts ) {

    update_rad_hydro_imex(
        compute_increment_hydro_explicit, compute_increment_rad_explicit,
        compute_increment_hydro_implicit, compute_increment_rad_implicit, dt,
        state, Grid, Basis, eos, opac, S_Limiter, opts );
  }

  /**
   * Fully coupled IMEX rad hydro update with SSPRK methods
   **/
  template <typename T, typename F, typename G, typename H>
  void update_rad_hydro_imex( T compute_increment_hydro_explicit,
                              F compute_increment_rad_explicit,
                              G compute_increment_hydro_implicit,
                              H compute_increment_rad_implicit, const Real dt,
                              State* state, GridStructure& Grid,
                              const ModalBasis* Basis, const EOS* eos,
                              const Opacity* opac, SlopeLimiter* S_Limiter,
                              const Options* opts ) {

    const auto& order = Basis->get_order( );
    const auto& ihi   = Grid.get_ihi( );

    auto uCF = state->get_u_cf( );
    auto uCR = state->get_u_cr( );

    Grid_s[0] = Grid;

    for ( unsigned short int iS = 0; iS < nStages; iS++ ) {
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 0 },
                                                  { ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iX, const int k ) {
            SumVar_U( 0, iX, k )   = uCF( 0, iX, k );
            SumVar_U( 1, iX, k )   = uCF( 1, iX, k );
            SumVar_U( 2, iX, k )   = uCF( 2, iX, k );
            SumVar_U_r( 0, iX, k ) = uCR( 0, iX, k );
            SumVar_U_r( 1, iX, k ) = uCR( 1, iX, k );
            StageData( iS, iX )    = Grid_s[iS].get_left_interface( iX );
          } );

      // --- Inner update loop ---

      for ( int j = 0; j < iS; j++ ) {
        const Real dt_a    = dt * explicit_tableau_.a_ij( iS, j );
        const Real dt_a_im = dt * implicit_tableau_.a_ij( iS, j );
        auto Us_j_h =
            Kokkos::subview( U_s, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto Us_j_r =
            Kokkos::subview( U_s_r, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto dUs_j_h =
            Kokkos::subview( dU_s, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto dUs_j_r =
            Kokkos::subview( dU_s_r, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto Flux_Uj = Kokkos::subview( Flux_U, j, Kokkos::ALL );

        compute_increment_rad_explicit( Us_j_r, Us_j_h, Grid_s[j], Basis, eos,
                                        dUs_j_r, Flux_q, dFlux_num, uCF_F_L,
                                        uCF_F_R, Flux_Uj, Flux_P, opts );
        compute_increment_hydro_explicit( Us_j_h, Us_j_r, Grid_s[j], Basis, eos,
                                          dUs_j_h, Flux_q, dFlux_num, uCF_F_L,
                                          uCF_F_R, Flux_Uj, Flux_P, opts );

        // inner sum
        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 0 },
                                                    { ihi + 2, order } ),
            KOKKOS_CLASS_LAMBDA( const int iX, const int k ) {
              SumVar_U( 0, iX, k ) += dt_a * dUs_j_h( 0, iX, k );
              SumVar_U( 1, iX, k ) += dt_a * dUs_j_h( 1, iX, k );
              SumVar_U( 2, iX, k ) += dt_a * dUs_j_h( 2, iX, k );
              SumVar_U_r( 0, iX, k ) += dt_a * dUs_j_r( 0, iX, k );
              SumVar_U_r( 1, iX, k ) += dt_a * dUs_j_r( 1, iX, k );
            } );

        Kokkos::parallel_for(
            "Timestepper :: implicit piece in inner loop",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 1 },
                                                    { ihi + 1, order } ),
            KOKKOS_CLASS_LAMBDA( const int iX, const int k ) {
              auto u_h =
                  Kokkos::subview( U_s, j, Kokkos::ALL, iX, Kokkos::ALL );
              auto u_r =
                  Kokkos::subview( U_s_r, j, Kokkos::ALL, iX, Kokkos::ALL );
              SumVar_U( 1, iX, k ) +=
                  dt_a_im * compute_increment_hydro_implicit( u_h, k, 1, u_r,
                                                              Grid_s[iS], Basis,
                                                              eos, opac, iX );
              SumVar_U( 2, iX, k ) +=
                  dt_a_im * compute_increment_hydro_implicit( u_h, k, 2, u_r,
                                                              Grid_s[iS], Basis,
                                                              eos, opac, iX );
              SumVar_U_r( 0, iX, k ) +=
                  dt_a_im * compute_increment_rad_implicit( u_r, k, 0, u_h,
                                                            Grid_s[iS], Basis,
                                                            eos, opac, iX );
              SumVar_U_r( 1, iX, k ) +=
                  dt_a_im * compute_increment_rad_implicit( u_r, k, 1, u_h,
                                                            Grid_s[iS], Basis,
                                                            eos, opac, iX );
            } );

        Kokkos::parallel_for(
            "Timestepper::StageData", ihi + 2,
            KOKKOS_CLASS_LAMBDA( const int iX ) {
              StageData( j, iX ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * Flux_Uj( iX );
            } );
      } // End inner loop

      auto StageDataj = Kokkos::subview( StageData, iS, Kokkos::ALL );
      Grid_s[iS].update_grid( StageDataj );

      // set U_s
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 0 },
                                                  { ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iX, const int k ) {
            U_s( iS, 0, iX, k )   = SumVar_U( 0, iX, k );
            U_s( iS, 1, iX, k )   = SumVar_U( 1, iX, k );
            U_s( iS, 2, iX, k )   = SumVar_U( 2, iX, k );
            U_s_r( iS, 0, iX, k ) = SumVar_U_r( 0, iX, k );
            U_s_r( iS, 1, iX, k ) = SumVar_U_r( 1, iX, k );
          } );

      // capturing sumvar_u_r bad?
      auto implicit_rad = [&]( View2D<Real> scratch, const int k, const int iC,
                               const View2D<Real> u_h, GridStructure& Grid,
                               const ModalBasis* Basis, const EOS* eos,
                               const Opacity* opac, const int iX ) {
        return SumVar_U_r( iC, iX, k ) +
               dt * implicit_tableau_.a_ij( iS, iS ) *
                   compute_increment_rad_implicit(
                       scratch, k, iC, u_h, Grid_s[iS], Basis, eos, opac, iX );
      };
      auto implicit_hydro = [&]( View2D<Real> scratch, const int k,
                                 const int iC, const View2D<Real> u_r,
                                 GridStructure& Grid, const ModalBasis* Basis,
                                 const EOS* eos, const Opacity* opac,
                                 const int iX ) {
        return SumVar_U( iC, iX, k ) +
               dt * implicit_tableau_.a_ij( iS, iS ) *
                   compute_increment_hydro_implicit(
                       scratch, k, iC, u_r, Grid_s[iS], Basis, eos, opac, iX );
      };

      // implicit update
      Kokkos::parallel_for(
          "Timestepper implicit",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 1, 1, 0 },
                                                  { 3, ihi + 1, order } ),
          KOKKOS_CLASS_LAMBDA( const int iC, const int iX, const int k ) {
            auto u_h = Kokkos::subview( U_s, iS, Kokkos::ALL, iX, Kokkos::ALL );
            auto u_r =
                Kokkos::subview( U_s_r, iS, Kokkos::ALL, iX, Kokkos::ALL );

            const Real temp2 = root_finders::fixed_point_aa(
                implicit_rad, k, u_r, iC - 1, u_h, Grid_s[iS], Basis, eos, opac,
                iX );
            U_s_r( iS, iC - 1, iX, k ) = temp2;

            const Real temp1 = root_finders::fixed_point_aa(
                implicit_hydro, k, u_h, iC, u_r, Grid_s[iS], Basis, eos, opac,
                iX );

            U_s( iS, iC, iX, k ) = temp1;
          } );

      // TODO(astrobarker): slope limit rad
      auto Us_j_h =
          Kokkos::subview( U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      apply_slope_limiter( S_Limiter, Us_j_h, &Grid_s[iS], Basis );
      auto Us_j_r =
          Kokkos::subview( U_s_r, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      apply_slope_limiter( S_Limiter, Us_j_r, &Grid_s[iS], Basis );
      bel::apply_bound_enforcing_limiter( Us_j_h, Basis, eos );
      bel::apply_bound_enforcing_limiter_rad( Us_j_r, Basis, eos );
    } // end outer loop

    for ( unsigned short int iS = 0; iS < nStages; iS++ ) {
      const Real dt_b    = dt * explicit_tableau_.b_i( iS );
      const Real dt_b_im = dt * implicit_tableau_.b_i( iS );
      auto Us_i_h =
          Kokkos::subview( U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto Us_i_r =
          Kokkos::subview( U_s_r, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto dUs_i_h =
          Kokkos::subview( dU_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto dUs_i_r =
          Kokkos::subview( dU_s_r, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto Flux_Ui = Kokkos::subview( Flux_U, iS, Kokkos::ALL );

      compute_increment_rad_explicit( Us_i_r, Us_i_h, Grid_s[iS], Basis, eos,
                                      dUs_i_r, Flux_q, dFlux_num, uCF_F_L,
                                      uCF_F_R, Flux_Ui, Flux_P, opts );
      compute_increment_hydro_explicit( Us_i_h, Us_i_r, Grid_s[iS], Basis, eos,
                                        dUs_i_h, Flux_q, dFlux_num, uCF_F_L,
                                        uCF_F_R, Flux_Ui, Flux_P, opts );
      Kokkos::parallel_for(
          "Timestepper :: u^(n+1) from the stages",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 0 },
                                                  { ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iX, const int k ) {
            auto u_h = Kokkos::subview( U_s, iS, Kokkos::ALL, iX, Kokkos::ALL );
            auto u_r =
                Kokkos::subview( U_s_r, iS, Kokkos::ALL, iX, Kokkos::ALL );
            uCF( 0, iX, k ) += dt_b * dUs_i_h( 0, iX, k );
            uCF( 1, iX, k ) += dt_b * dUs_i_h( 1, iX, k );
            uCF( 2, iX, k ) += dt_b * dUs_i_h( 2, iX, k );

            uCF( 1, iX, k ) += dt_b_im * compute_increment_hydro_implicit(
                                             u_h, k, 1, u_r, Grid_s[iS], Basis,
                                             eos, opac, iX );
            uCF( 2, iX, k ) += dt_b_im * compute_increment_hydro_implicit(
                                             u_h, k, 2, u_r, Grid_s[iS], Basis,
                                             eos, opac, iX );

            uCR( 0, iX, k ) += dt_b * dUs_i_r( 0, iX, k );
            uCR( 1, iX, k ) += dt_b * dUs_i_r( 1, iX, k );

            uCR( 0, iX, k ) += dt_b_im * compute_increment_rad_implicit(
                                             u_r, k, 0, u_h, Grid_s[iS], Basis,
                                             eos, opac, iX );
            uCR( 1, iX, k ) += dt_b_im * compute_increment_rad_implicit(
                                             u_r, k, 1, u_h, Grid_s[iS], Basis,
                                             eos, opac, iX );
          } );

      Kokkos::parallel_for(
          "Timestepper::StageData::final", ihi + 2,
          KOKKOS_CLASS_LAMBDA( const int iX ) {
            StageData( iS, iX ) += dt_b * Flux_Ui( iX );
          } );
      auto StageDataj = Kokkos::subview( StageData, iS, Kokkos::ALL );
      Grid_s[iS].update_grid( StageDataj );
    }

    // TODO(astrobarker): slope limit rad
    Grid = Grid_s[nStages - 1];
    apply_slope_limiter( S_Limiter, uCF, &Grid, Basis );
    apply_slope_limiter( S_Limiter, uCR, &Grid, Basis );
    bel::apply_bound_enforcing_limiter( uCF, Basis, eos );
    bel::apply_bound_enforcing_limiter_rad( uCR, Basis, eos );
  }

 private:
  const int mSize;
  const int nStages;
  const int tOrder;
  const std::string BC;

  // tableaus
  // TODO(astrobarker): always have both tableaus?
  // Maybe create an IMEX class... (or new implicit and explicit classes)
  // View2D<Real> a_ij;
  // View2D<Real> b_ij;
  ButcherTableau implicit_tableau_;
  ButcherTableau explicit_tableau_;

  // Hold stage data
  View4D<Real> U_s{ };
  View4D<Real> dU_s{ };
  View4D<Real> U_s_r{ };
  View4D<Real> dU_s_r{ };
  View3D<Real> SumVar_U{ };
  View3D<Real> SumVar_U_r{ };
  std::vector<GridStructure> Grid_s{ };

  // StageData Holds cell left interface positions
  View2D<Real> StageData{ };

  // Variables to pass to update step
  View3D<Real> Flux_q{ };

  View2D<Real> dFlux_num{ };
  View2D<Real> uCF_F_L{ };
  View2D<Real> uCF_F_R{ };
  View2D<Real> Flux_U{ };
  View1D<Real> Flux_P{ };
};

#endif // TIMESTEPPER_HPP_
