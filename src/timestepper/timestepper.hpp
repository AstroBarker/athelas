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
 *
 * TODO(astrobaker) move to calling step<fluid> / step<radhydro>
 */

#include "abstractions.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "bound_enforcing_limiter.hpp"
#include "eos_variant.hpp"
#include "fluid/fluid_discretization.hpp"
#include "opacity/opac_variant.hpp"
#include "polynomial_basis.hpp"
#include "problem_in.hpp"
#include "radiation/rad_discretization.hpp"
#include "slope_limiter.hpp"
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
    update_fluid_explicit( dt, state, grid, fluid_basis, eos, S_Limiter, opts,
                           bcs );
  }

  /**
   * Explicit fluid update with SSPRK methods
   **/
  void update_fluid_explicit( const double dt, State* state,
                              GridStructure& grid,
                              const ModalBasis* fluid_basis, const EOS* eos,
                              SlopeLimiter* S_Limiter, const Options* opts,
                              BoundaryConditions* bcs ) {

    const auto& order = fluid_basis->get_order( );
    const auto& ihi   = grid.get_ihi( );

    auto U = state->get_u_cf( );

    const int nvars = U.extent( 0 );

    grid_s_[0] = grid;

    for ( int iS = 0; iS < nStages_; ++iS ) {
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
        compute_increment_fluid_explicit( Us_j, grid_s_[j], fluid_basis, eos,
                                          dUs_j, dFlux_num_, uCF_F_L_, uCF_F_R_,
                                          flux_u_j, opts, bcs );

        // inner sum
        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                    { nvars, ihi + 2, order } ),
            KOKKOS_CLASS_LAMBDA( const int iCF, const int iX, const int k ) {
              SumVar_U_( iCF, iX, k ) +=
                  dt * integrator_.explicit_tableau.a_ij( iS, j ) *
                  dUs_j( iCF, iX, k );
            } );

        Kokkos::parallel_for(
            "Timestepper::stage_data_", ihi + 2,
            KOKKOS_CLASS_LAMBDA( const int iX ) {
              stage_data_( iS, iX ) +=
                  dt * integrator_.explicit_tableau.a_ij( iS, j ) *
                  flux_u_j( iX );
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
      // apply_slope_limiter( S_Limiter, Us_j, &grid_s_[iS], fluid_basis, eos );
      bel::apply_bound_enforcing_limiter( Us_j, fluid_basis, eos );
    } // end outer loop

    for ( int iS = 0; iS < nStages_; ++iS ) {
      auto Us_j =
          Kokkos::subview( U_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto dUs_j =
          Kokkos::subview( dU_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto flux_u_j = Kokkos::subview( flux_u_, iS, Kokkos::ALL );

      compute_increment_fluid_explicit( Us_j, grid_s_[iS], fluid_basis, eos,
                                        dUs_j, dFlux_num_, uCF_F_L_, uCF_F_R_,
                                        flux_u_j, opts, bcs );
      Kokkos::parallel_for(
          "Timestepper :: u^(n+1) from the stages",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { nvars, ihi + 2, order } ),
          KOKKOS_CLASS_LAMBDA( const int iCF, const int iX, const int k ) {
            U( iCF, iX, k ) += dt * integrator_.explicit_tableau.b_i( iS ) *
                               dUs_j( iCF, iX, k );
          } );

      Kokkos::parallel_for(
          "Timestepper::stage_data_::final", ihi + 2,
          KOKKOS_CLASS_LAMBDA( const int iX ) {
            stage_data_( 0, iX ) +=
                dt * flux_u_j( iX ) * integrator_.explicit_tableau.b_i( iS );
          } );
      auto stage_data_j = Kokkos::subview( stage_data_, 0, Kokkos::ALL );
      grid_s_[iS].update_grid( stage_data_j );
    }

    grid = grid_s_[nStages_ - 1];
    apply_slope_limiter( S_Limiter, U, &grid, fluid_basis, eos );
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

    update_rad_hydro_imex( dt, state, grid, fluid_basis, rad_basis, eos, opac,
                           S_Limiter, opts, bcs );
  }

  /**
   * Fully coupled IMEX rad hydro update with SSPRK methods
   **/
  void update_rad_hydro_imex( const double dt, State* state,
                              GridStructure& grid,
                              const ModalBasis* fluid_basis,
                              const ModalBasis* rad_basis, const EOS* eos,
                              const Opacity* opac, SlopeLimiter* S_Limiter,
                              const Options* opts, BoundaryConditions* bcs ) {

    const auto& order = fluid_basis->get_order( );
    const auto& ihi   = grid.get_ihi( );

    auto uCF = state->get_u_cf( );
    auto uCR = state->get_u_cr( );

    grid_s_[0] = grid;

    for ( int iS = 0; iS < nStages_; ++iS ) {
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
        const double dt_a    = dt * integrator_.explicit_tableau.a_ij( iS, j );
        const double dt_a_im = dt * integrator_.implicit_tableau.a_ij( iS, j );
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
            Us_j_r, Us_j_h, grid_s_[j], rad_basis, fluid_basis, eos, dUs_j_r,
            dFlux_num_, uCF_F_L_, uCF_F_R_, flux_u_j, opts, bcs );
        compute_increment_fluid_explicit( // hydro
            Us_j_h, grid_s_[j], fluid_basis, eos, dUs_j_h, dFlux_num_, uCF_F_L_,
            uCF_F_R_, flux_u_j, opts, bcs );

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
                  dt_a_im * compute_increment_fluid_source(
                                u_h, k, 1, u_r, grid_s_[iS], fluid_basis,
                                rad_basis, eos, opac, iX );
              SumVar_U_( 2, iX, k ) +=
                  dt_a_im * compute_increment_fluid_source(
                                u_h, k, 2, u_r, grid_s_[iS], fluid_basis,
                                rad_basis, eos, opac, iX );
              SumVar_U_r_( 0, iX, k ) +=
                  dt_a_im * compute_increment_rad_source(
                                u_r, k, 0, u_h, grid_s_[iS], fluid_basis,
                                rad_basis, eos, opac, iX );
              SumVar_U_r_( 1, iX, k ) +=
                  dt_a_im * compute_increment_rad_source(
                                u_r, k, 1, u_h, grid_s_[iS], fluid_basis,
                                rad_basis, eos, opac, iX );
            } );

        Kokkos::parallel_for(
            "Timestepper::stage_data_", ihi + 2,
            KOKKOS_CLASS_LAMBDA( const int iX ) {
              stage_data_( j, iX ) +=
                  dt * integrator_.explicit_tableau.a_ij( iS, j ) *
                  flux_u_j( iX );
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

      auto Us_j_h =
          Kokkos::subview( U_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      auto Us_j_r =
          Kokkos::subview( U_s_r_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      apply_slope_limiter( S_Limiter, Us_j_h, &grid_s_[iS], fluid_basis, eos );
      apply_slope_limiter( S_Limiter, Us_j_r, &grid_s_[iS], rad_basis, eos );
      bel::apply_bound_enforcing_limiter( Us_j_h, fluid_basis, eos );
      bel::apply_bound_enforcing_limiter_rad( Us_j_r, rad_basis, eos );
      bel::apply_bound_enforcing_limiter_rad( SumVar_U_r_, rad_basis, eos );
      bel::apply_bound_enforcing_limiter( SumVar_U_, rad_basis, eos );

      // implicit update
      // TODO(astrobarker) cleanup scratch mess
      // TODO(astrobarker) combined U, U_r
      View3D<double> scratch_implicit( "scratch_implicit", ihi + 2, 5, order );
      View3D<double> scratch_solution_k( "scratch_solution_k", ihi + 2, 5,
                                         order );
      View3D<double> scratch_solution_km1( "scratch_solution_km1", ihi + 2, 5,
                                           order );
      View3D<double> R( "R", ihi + 2, 5, order );
      Kokkos::parallel_for(
          "Timestepper implicit", Kokkos::RangePolicy<>( 1, ihi + 1 ),
          KOKKOS_CLASS_LAMBDA( const int iX ) {
            const double dt_a_ii =
                dt * integrator_.implicit_tableau.a_ij( iS, iS );
            auto u_h =
                Kokkos::subview( U_s_, iS, Kokkos::ALL, iX, Kokkos::ALL );
            auto u_r =
                Kokkos::subview( U_s_r_, iS, Kokkos::ALL, iX, Kokkos::ALL );
            auto scratch_sol_ix     = Kokkos::subview( scratch_implicit, iX,
                                                       Kokkos::ALL, Kokkos::ALL );
            auto scratch_sol_ix_k   = Kokkos::subview( scratch_solution_k, iX,
                                                       Kokkos::ALL, Kokkos::ALL );
            auto scratch_sol_ix_km1 = Kokkos::subview(
                scratch_solution_km1, iX, Kokkos::ALL, Kokkos::ALL );
            auto R_ix = Kokkos::subview( R, iX, Kokkos::ALL, Kokkos::ALL );

            // TODO(astrobarker): invert loops
            for ( int k = 0; k < order; ++k ) {
              // set hydro vars
              for ( int i = 0; i < 3; ++i ) {
                scratch_sol_ix_k( i, k )   = u_h( i, k );
                scratch_sol_ix_km1( i, k ) = u_h( i, k );
                scratch_sol_ix( i, k )     = u_h( i, k );
                R_ix( i, k )               = SumVar_U_( i, iX, k );
              }
              // set rad vars
              for ( int i = 3; i < 5; ++i ) {
                scratch_sol_ix_k( i, k )   = u_r( i - 3, k );
                scratch_sol_ix_km1( i, k ) = u_r( i - 3, k );
                scratch_sol_ix( i, k )     = u_r( i - 3, k );
                R_ix( i, k )               = SumVar_U_r_( i - 3, iX, k );
              }
            }

            root_finders::fixed_point_radhydro(
                R_ix, dt_a_ii, scratch_sol_ix_k, scratch_sol_ix_km1,
                scratch_sol_ix, grid_s_[iS], fluid_basis, rad_basis, eos, opac,
                iX );

            // TODO(astrobarker): invert loops
            for ( int k = 0; k < order; ++k ) {
              for ( int i = 3; i < 5; ++i ) {
                U_s_r_( iS, i - 3, iX, k ) = scratch_sol_ix( i, k );
              }
              // hydro (no need to update density)
              for ( int i = 1; i < 3; ++i ) {
                U_s_( iS, i, iX, k ) = scratch_sol_ix( i, k );
              }
            }
          } );

      // TODO(astrobarker): slope limit rad
      apply_slope_limiter( S_Limiter, Us_j_h, &grid_s_[iS], fluid_basis, eos );
      apply_slope_limiter( S_Limiter, Us_j_r, &grid_s_[iS], rad_basis, eos );
      bel::apply_bound_enforcing_limiter( Us_j_h, fluid_basis, eos );
      bel::apply_bound_enforcing_limiter_rad( Us_j_r, rad_basis, eos );
    } // end outer loop

    for ( int iS = 0; iS < nStages_; ++iS ) {
      const double dt_b    = dt * integrator_.explicit_tableau.b_i( iS );
      const double dt_b_im = dt * integrator_.implicit_tableau.b_i( iS );
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
                                      fluid_basis, eos, dUs_i_r, dFlux_num_,
                                      uCF_F_L_, uCF_F_R_, flux_u_i, opts, bcs );
      compute_increment_fluid_explicit( Us_i_h, grid_s_[iS], fluid_basis, eos,
                                        dUs_i_h, dFlux_num_, uCF_F_L_, uCF_F_R_,
                                        flux_u_i, opts, bcs );
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

            uCF( 1, iX, k ) +=
                dt_b_im * compute_increment_fluid_source(
                              u_h, k, 1, u_r, grid_s_[iS], fluid_basis,
                              rad_basis, eos, opac, iX );
            uCF( 2, iX, k ) +=
                dt_b_im * compute_increment_fluid_source(
                              u_h, k, 2, u_r, grid_s_[iS], fluid_basis,
                              rad_basis, eos, opac, iX );

            uCR( 0, iX, k ) += dt_b * dUs_i_r( 0, iX, k );
            uCR( 1, iX, k ) += dt_b * dUs_i_r( 1, iX, k );

            uCR( 0, iX, k ) +=
                dt_b_im * compute_increment_rad_source(
                              u_r, k, 0, u_h, grid_s_[iS], fluid_basis,
                              rad_basis, eos, opac, iX );
            uCR( 1, iX, k ) +=
                dt_b_im * compute_increment_rad_source(
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
  int mSize_;

  // tableaus
  RKIntegrator integrator_;

  int nStages_;
  int tOrder_;

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

  View2D<double> dFlux_num_{ };
  View2D<double> uCF_F_L_{ };
  View2D<double> uCF_F_R_{ };
  View2D<double> flux_u_{ };
};
