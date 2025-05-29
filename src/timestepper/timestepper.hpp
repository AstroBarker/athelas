#pragma once
/**
 * @file timestepper.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Primary time marching routine.
 *
 * @details Timestppers for hydro and rad hydro.
 *          Uses explicit for transport terms and implicit for coupling.
 */

#include <math.h>

#include "abstractions.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "bound_enforcing_limiter.hpp"
#include "eos.hpp"
#include "fluid/fluid_discretization.hpp"
#include "opacity/opac.hpp"
#include "opacity/opac_variant.hpp"
#include "polynomial_basis.hpp"
#include "problem_in.hpp"
#include "radiation/rad_discretization.hpp"
#include "slope_limiter.hpp"
#include "slope_limiter_base.hpp"
#include "solvers/root_finders.hpp"
#include "state.hpp"
#include "tableau.hpp"

using bc::BoundaryConditions;
using fluid::compute_increment_fluid_explicit;
using fluid::compute_increment_fluid_source;
using radiation::compute_increment_rad_explicit;
using radiation::compute_increment_rad_source;

class TimeStepper {

 public:
  // TODO(astrobarker): Is it possible to initialize grid_s_ from grid directly?
  TimeStepper( const ProblemIn* pin, GridStructure* grid );

  void initialize_timestepper( );

  /**
   * Update fluid solution with SSPRK methods
   **/
  void update_fluid( const double dt, State* state, GridStructure& grid,
                     const ModalBasis* fluid_basis, const EOS* eos,
                     SlopeLimiter* S_Limiter, const Options* opts,
                     BoundaryConditions* bcs ) {

    // hydro explicit update
    update_fluid_explicit( dt, state, grid, fluid_basis, eos, S_Limiter, opts, bcs );
  }

  /**
   * Explicit fluid update with SSPRK methods
   **/
  void update_fluid_explicit( const double dt, State* state, GridStructure& grid,
                              const ModalBasis* fluid_basis, const EOS* eos,
                              SlopeLimiter* S_Limiter, const Options* opts,
                              BoundaryConditions* bcs ) {

    const auto& order = fluid_basis->get_order( );
    const auto& ihi   = grid.get_ihi( );

    auto U   = state->get_u_cf( );

    const int nvars = U.extent( 0 );

    grid_s_[0] = grid;

    for ( unsigned short int iS = 0; iS < nStages_; iS++ ) {
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { nvars, ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iCF, const int iX, const int k ) {
            SumVar_U_( iCF, iX, k ) = U( iCF, iX, k );
            stage_data_( iS, iX )   = grid.get_left_interface( iX );
            // stage_data_( iS, iX )    = grid_s_[iS].get_left_interface( iX );
          } );

      // --- Inner update loop ---

      for ( int j = 0; j < iS; j++ ) {
        auto Us_j =
            Kokkos::subview( U_s_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto dUs_j =
            Kokkos::subview( dU_s_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto flux_u_j = Kokkos::subview( flux_u_, j, Kokkos::ALL );
        compute_increment_fluid_explicit(
            Us_j, grid_s_[j], fluid_basis, eos, dUs_j, flux_q_, dFlux_num_,
            uCF_F_L_, uCF_F_R_, flux_u_j, flux_p_, opts, bcs );

        // inner sum
        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                    { nvars, ihi + 2, order } ),
            KOKKOS_CLASS_LAMBDA( const int iCF, const int iX, const int k ) {
              SumVar_U_( iCF, iX, k ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * dUs_j( iCF, iX, k );
            } );

        Kokkos::parallel_for(
            "Timestepper::stage_data_", ihi + 2,
            KOKKOS_CLASS_LAMBDA( const int iX ) {
              stage_data_( iS, iX ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * flux_u_j( iX );
            } );
      } // End inner loop

      // set U_s
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { nvars, ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iCF, const int iX, const int k ) {
            U_s_( iS, iCF, iX, k ) = SumVar_U_( iCF, iX, k );
          } );

      auto stage_data_j = Kokkos::subview( stage_data_, iS, Kokkos::ALL );
      grid_s_[iS].update_grid( stage_data_j );

      auto Us_j =
          Kokkos::subview( U_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      apply_slope_limiter( S_Limiter, Us_j, &grid_s_[iS], fluid_basis, eos );
      bel::apply_bound_enforcing_limiter( Us_j, fluid_basis, eos );
    } // end outer loop

    for ( unsigned short int iS = 0; iS < nStages_; iS++ ) {
      auto Us_j =
          Kokkos::subview( U_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto dUs_j =
          Kokkos::subview( dU_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto flux_u_j = Kokkos::subview( flux_u_, iS, Kokkos::ALL );

      compute_increment_fluid_explicit(
          Us_j, grid_s_[iS], fluid_basis, eos, dUs_j, flux_q_, dFlux_num_,
          uCF_F_L_, uCF_F_R_, flux_u_j, flux_p_, opts, bcs );
      Kokkos::parallel_for(
          "Timestepper :: u^(n+1) from the stages",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { nvars, ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iCF, const int iX, const int k ) {
            U( iCF, iX, k ) +=
                dt * explicit_tableau_.b_i( iS ) * dUs_j( iCF, iX, k );
          } );

      Kokkos::parallel_for(
          "Timestepper::stage_data_::final", ihi + 2,
          KOKKOS_CLASS_LAMBDA( const int iX ) {
            stage_data_( 0, iX ) +=
                dt * flux_u_j( iX ) * explicit_tableau_.b_i( iS );
          } );
      auto stage_data_j = Kokkos::subview( stage_data_, 0, Kokkos::ALL );
      grid_s_[iS].update_grid( stage_data_j );
    }

    grid = grid_s_[nStages_ - 1];
    // apply_slope_limiter( S_Limiter, U, &grid, fluid_basis, eos );
    bel::apply_bound_enforcing_limiter( U, fluid_basis, eos );
  }

  /**
   * Update rad hydro solution with SSPRK methods
   **/
  void update_rad_hydro( const double dt, State* state, GridStructure& grid,
                         const ModalBasis* fluid_basis, 
                         const ModalBasis* rad_basis, const EOS* eos,
                         const Opacity* opac, SlopeLimiter* S_Limiter,
                         const Options* opts, BoundaryConditions* bcs ) {

    update_rad_hydro_imex( dt, state, grid, fluid_basis, rad_basis, eos, opac, S_Limiter, opts,
                           bcs );
  }

  /**
   * Fully coupled IMEX rad hydro update with SSPRK methods
   **/
  void update_rad_hydro_imex( const double dt, State* state, GridStructure& grid,
                              const ModalBasis* fluid_basis, 
                              const ModalBasis* rad_basis, const EOS* eos,
                              const Opacity* opac, SlopeLimiter* S_Limiter,
                              const Options* opts, BoundaryConditions* bcs ) {

    const auto& order = fluid_basis->get_order( );
    const auto& ihi   = grid.get_ihi( );

    auto uCF = state->get_u_cf( );
    auto uCR = state->get_u_cr( );

    grid_s_[0] = grid;

    for ( unsigned short int iS = 0; iS < nStages_; iS++ ) {
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 0 },
                                                  { ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iX, const int k ) {
            SumVar_U_( 0, iX, k )   = uCF( 0, iX, k );
            SumVar_U_( 1, iX, k )   = uCF( 1, iX, k );
            SumVar_U_( 2, iX, k )   = uCF( 2, iX, k );
            SumVar_U_r_( 0, iX, k ) = uCR( 0, iX, k );
            SumVar_U_r_( 1, iX, k ) = uCR( 1, iX, k );
            stage_data_( iS, iX )   = grid_s_[iS].get_left_interface( iX );
          } );

      // --- Inner update loop ---

      for ( int j = 0; j < iS; j++ ) {
        const double dt_a    = dt * explicit_tableau_.a_ij( iS, j );
        const double dt_a_im = dt * implicit_tableau_.a_ij( iS, j );
        auto Us_j_h =
            Kokkos::subview( U_s_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto Us_j_r =
            Kokkos::subview( U_s_r_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto dUs_j_h =
            Kokkos::subview( dU_s_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto dUs_j_r  = Kokkos::subview( dU_s_r_, j, Kokkos::ALL, Kokkos::ALL,
                                         Kokkos::ALL );
        auto flux_u_j = Kokkos::subview( flux_u_, j, Kokkos::ALL );

        compute_increment_rad_explicit( // rad
            Us_j_r, Us_j_h, grid_s_[j], rad_basis, fluid_basis, eos, dUs_j_r, flux_q_,
            dFlux_num_, uCF_F_L_, uCF_F_R_, flux_u_j, flux_p_, opts, bcs );
        compute_increment_fluid_explicit( // hydro
            Us_j_h, grid_s_[j], fluid_basis, eos, dUs_j_h, flux_q_,
            dFlux_num_, uCF_F_L_, uCF_F_R_, flux_u_j, flux_p_, opts, bcs );

        // inner sum
        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 0 },
                                                    { ihi + 2, order } ),
            KOKKOS_CLASS_LAMBDA( const int iX, const int k ) {
              SumVar_U_( 0, iX, k ) += dt_a * dUs_j_h( 0, iX, k );
              SumVar_U_( 1, iX, k ) += dt_a * dUs_j_h( 1, iX, k );
              SumVar_U_( 2, iX, k ) += dt_a * dUs_j_h( 2, iX, k );
              SumVar_U_r_( 0, iX, k ) += dt_a * dUs_j_r( 0, iX, k );
              SumVar_U_r_( 1, iX, k ) += dt_a * dUs_j_r( 1, iX, k );
            } );

        Kokkos::parallel_for(
            "Timestepper :: implicit piece in inner loop",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 1, 0 },
                                                    { ihi + 1, order } ),
            KOKKOS_CLASS_LAMBDA( const int iX, const int k ) {
              auto u_h =
                  Kokkos::subview( U_s_, j, Kokkos::ALL, iX, Kokkos::ALL );
              auto u_r =
                  Kokkos::subview( U_s_r_, j, Kokkos::ALL, iX, Kokkos::ALL );
              SumVar_U_( 1, iX, k ) +=
                  dt_a_im * compute_increment_fluid_source( u_h, k, 1, u_r,
                                                            grid_s_[iS], fluid_basis,
                                                            rad_basis, eos, opac, iX );
              SumVar_U_( 2, iX, k ) +=
                  dt_a_im * compute_increment_fluid_source( u_h, k, 2, u_r,
                                                            grid_s_[iS], fluid_basis,
                                                            rad_basis, eos, opac, iX );
              SumVar_U_r_( 0, iX, k ) +=
                  dt_a_im * compute_increment_rad_source( u_r, k, 0, u_h,
                                                          grid_s_[iS], fluid_basis,
                                                          rad_basis, eos, opac, iX );
              SumVar_U_r_( 1, iX, k ) +=
                  dt_a_im * compute_increment_rad_source( u_r, k, 1, u_h,
                                                          grid_s_[iS], fluid_basis,
                                                          rad_basis, eos, opac, iX );
            } );

        Kokkos::parallel_for(
            "Timestepper::stage_data_", ihi + 2,
            KOKKOS_CLASS_LAMBDA( const int iX ) {
              stage_data_( j, iX ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * flux_u_j( iX );
            } );
      } // End inner loop

      auto stage_data_j = Kokkos::subview( stage_data_, iS, Kokkos::ALL );
      grid_s_[iS].update_grid( stage_data_j );

      // set U_s
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 0 },
                                                  { ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iX, const int k ) {
            U_s_( iS, 0, iX, k )   = SumVar_U_( 0, iX, k );
            U_s_( iS, 1, iX, k )   = SumVar_U_( 1, iX, k );
            U_s_( iS, 2, iX, k )   = SumVar_U_( 2, iX, k );
            U_s_r_( iS, 0, iX, k ) = SumVar_U_r_( 0, iX, k );
            U_s_r_( iS, 1, iX, k ) = SumVar_U_r_( 1, iX, k );
          } );

      // capturing sumvar_u_r bad?
      auto implicit_rad = [&]( View2D<double> scratch, const int k, const int iC,
                               const View2D<double> u_h, GridStructure& grid,
                               const ModalBasis* fluid_basis, 
                               const ModalBasis* rad_basis, const EOS* eos,
                               const Opacity* opac, const int iX ) {
        return SumVar_U_r_( iC, iX, k ) +
               dt * implicit_tableau_.a_ij( iS, iS ) *
                   compute_increment_rad_source(
                       scratch, k, iC, u_h, grid_s_[iS], fluid_basis, rad_basis, eos, opac, iX );
      };
      auto implicit_hydro = [&]( View2D<double> scratch, const int k,
                                 const int iC, const View2D<double> u_r,
                                 GridStructure& grid, const ModalBasis* fluid_basis,
                                 const ModalBasis* rad_basis, const EOS* eos, 
                                 const Opacity* opac, const int iX ) {
        return SumVar_U_( iC, iX, k ) +
               dt * implicit_tableau_.a_ij( iS, iS ) *
                   compute_increment_fluid_source(
                       scratch, k, iC, u_r, grid_s_[iS], fluid_basis, rad_basis, eos, opac, iX );
      };

      // implicit update
      Kokkos::parallel_for(
          "Timestepper implicit",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 1, 1, 0 },
                                                  { 3, ihi + 1, order } ),
          KOKKOS_CLASS_LAMBDA( const int iC, const int iX, const int k ) {
            auto u_h =
                Kokkos::subview( U_s_, iS, Kokkos::ALL, iX, Kokkos::ALL );
            auto u_r =
                Kokkos::subview( U_s_r_, iS, Kokkos::ALL, iX, Kokkos::ALL );

            const double temp2 = root_finders::fixed_point_aa(
                implicit_rad, k, u_r, iC - 1, u_h, grid_s_[iS], fluid_basis, rad_basis, eos,
                opac, iX );
            U_s_r_( iS, iC - 1, iX, k ) = temp2;

            const double temp1 = root_finders::fixed_point_aa(
                implicit_hydro, k, u_h, iC, u_r, grid_s_[iS], fluid_basis, rad_basis, eos, opac,
                iX );

            U_s_( iS, iC, iX, k ) = temp1;
          } );

      // TODO(astrobarker): slope limit rad
      auto Us_j_h =
          Kokkos::subview( U_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      apply_slope_limiter( S_Limiter, Us_j_h, &grid_s_[iS], fluid_basis, eos );
      auto Us_j_r =
          Kokkos::subview( U_s_r_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      apply_slope_limiter( S_Limiter, Us_j_r, &grid_s_[iS], rad_basis, eos );
      bel::apply_bound_enforcing_limiter( Us_j_h, fluid_basis, eos );
      bel::apply_bound_enforcing_limiter_rad( Us_j_r, rad_basis, eos );
    } // end outer loop

    for ( unsigned short int iS = 0; iS < nStages_; iS++ ) {
      const double dt_b    = dt * explicit_tableau_.b_i( iS );
      const double dt_b_im = dt * implicit_tableau_.b_i( iS );
      auto Us_i_h =
          Kokkos::subview( U_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto Us_i_r =
          Kokkos::subview( U_s_r_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto dUs_i_h =
          Kokkos::subview( dU_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto dUs_i_r =
          Kokkos::subview( dU_s_r_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto flux_u_i = Kokkos::subview( flux_u_, iS, Kokkos::ALL );

      compute_increment_rad_explicit( Us_i_r, Us_i_h, grid_s_[iS], rad_basis, 
                                      fluid_basis, eos, dUs_i_r, flux_q_, 
                                      dFlux_num_, uCF_F_L_, uCF_F_R_, flux_u_i,
                                      flux_p_, opts, bcs );
      compute_increment_fluid_explicit(
          Us_i_h, grid_s_[iS], fluid_basis, eos, dUs_i_h, flux_q_, dFlux_num_,
          uCF_F_L_, uCF_F_R_, flux_u_i, flux_p_, opts, bcs );
      Kokkos::parallel_for(
          "Timestepper :: u^(n+1) from the stages",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 0 },
                                                  { ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iX, const int k ) {
            auto u_h =
                Kokkos::subview( U_s_, iS, Kokkos::ALL, iX, Kokkos::ALL );
            auto u_r =
                Kokkos::subview( U_s_r_, iS, Kokkos::ALL, iX, Kokkos::ALL );
            uCF( 0, iX, k ) += dt_b * dUs_i_h( 0, iX, k );
            uCF( 1, iX, k ) += dt_b * dUs_i_h( 1, iX, k );
            uCF( 2, iX, k ) += dt_b * dUs_i_h( 2, iX, k );

            uCF( 1, iX, k ) += dt_b_im * compute_increment_fluid_source(
                                             u_h, k, 1, u_r, grid_s_[iS], fluid_basis,
                                             rad_basis, eos, opac, iX );
            uCF( 2, iX, k ) += dt_b_im * compute_increment_fluid_source(
                                             u_h, k, 2, u_r, grid_s_[iS], fluid_basis,
                                             rad_basis, eos, opac, iX );

            uCR( 0, iX, k ) += dt_b * dUs_i_r( 0, iX, k );
            uCR( 1, iX, k ) += dt_b * dUs_i_r( 1, iX, k );

            uCR( 0, iX, k ) += dt_b_im * compute_increment_rad_source(
                                             u_r, k, 0, u_h, grid_s_[iS], fluid_basis,
                                             rad_basis, eos, opac, iX );
            uCR( 1, iX, k ) += dt_b_im * compute_increment_rad_source(
                                             u_r, k, 1, u_h, grid_s_[iS], fluid_basis,
                                             rad_basis, eos, opac, iX );
          } );

      Kokkos::parallel_for(
          "Timestepper::stage_data_::final", ihi + 2,
          KOKKOS_CLASS_LAMBDA( const int iX ) {
            stage_data_( iS, iX ) += dt_b * flux_u_i( iX );
          } );
      auto stage_data_j = Kokkos::subview( stage_data_, iS, Kokkos::ALL );
      grid_s_[iS].update_grid( stage_data_j );
    }

    // TODO(astrobarker): slope limit rad
    grid = grid_s_[nStages_ - 1];
    apply_slope_limiter( S_Limiter, uCF, &grid, fluid_basis, eos );
    apply_slope_limiter( S_Limiter, uCR, &grid, rad_basis, eos );
    bel::apply_bound_enforcing_limiter( uCF, fluid_basis, eos );
    bel::apply_bound_enforcing_limiter_rad( uCR, rad_basis, eos );
  }

 private:
  const int mSize_;
  const int nStages_;
  const int tOrder_;

  // tableaus
  // TODO(astrobarker): always have both tableaus?
  // Maybe create an IMEX class... (or new implicit and explicit classes)
  ButcherTableau implicit_tableau_;
  ButcherTableau explicit_tableau_;

  // Hold stage data
  View4D<double> U_s_{ };
  View4D<double> dU_s_{ };
  View4D<double> U_s_r_{ };
  View4D<double> dU_s_r_{ };
  View3D<double> SumVar_U_{ };
  View3D<double> SumVar_U_r_{ };
  std::vector<GridStructure> grid_s_{ };

  // stage_data_ Holds cell left interface positions
  View2D<double> stage_data_{ };

  // Variables to pass to update step
  View3D<double> flux_q_{ };

  View2D<double> dFlux_num_{ };
  View2D<double> uCF_F_L_{ };
  View2D<double> uCF_F_R_{ };
  View2D<double> flux_u_{ };
  View1D<double> flux_p_{ };
};
