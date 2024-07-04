#ifndef TIMESTEPPER_HPP_
#define TIMESTEPPER_HPP_

/**
 * File     :  timestepper.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for SSPRK timestepping
 **/

#include "abstractions.hpp"
#include "bound_enforcing_limiter.hpp"
#include "eos.hpp"
#include "polynomial_basis.hpp"
#include "problem_in.hpp"
#include "slope_limiter.hpp"
#include "state.hpp"
#include "tableau.hpp"

class TimeStepper {

 public:
  // TODO: Is it possible to initialize Grid_s from Grid directly?
  TimeStepper( ProblemIn *pin, GridStructure &Grid );

  void InitializeTimestepper( );

  /**
   * Update fluid solution with SSPRK methods
   **/
  template <typename T>
  void UpdateFluid( T ComputeIncrement, const Real dt, State *state,
                    GridStructure &Grid, const ModalBasis *Basis,
                    const EOS *eos, SlopeLimiter *S_Limiter,
                    const Options opts ) {

    // hydro explicity update
    UpdateFluid_Explicit( ComputeIncrement, dt, state, Grid, Basis, eos,
                          S_Limiter, opts );
  }

  /**
   * Explicit fluid update with SSPRK methods
   **/
  template <typename T>
  void UpdateFluid_Explicit( T ComputeIncrement, const Real dt, State *state,
                             GridStructure &Grid, const ModalBasis *Basis,
                             const EOS *eos, SlopeLimiter *S_Limiter,
                             const Options opts ) {

    const auto &order = Basis->Get_Order( );
    const auto &ihi   = Grid.Get_ihi( );

    auto U   = state->Get_uCF( );
    auto uCR = state->Get_uCR( );

    const int nvars = U.extent( 0 );

    Grid_s[0] = Grid;

    for ( unsigned short int iS = 0; iS < nStages; iS++ ) {
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { order, ihi + 2, nvars } ),
          KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
            SumVar_U( iCF, iX, k ) = U( iCF, iX, k );
            StageData( iS, iX )    = Grid_s[iS].Get_LeftInterface( iX );
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
                                                    { order, ihi + 2, nvars } ),
            KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
              SumVar_U( iCF, iX, k ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * dUs_j( iCF, iX, k );
            } );

        Kokkos::parallel_for(
            "Timestepper::StageData", ihi + 2, KOKKOS_LAMBDA( const int iX ) {
              StageData( iS, iX ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * Flux_Uj( iX );
            } );
      } // End inner loop

      // set U_s
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { order, ihi + 2, nvars } ),
          KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
            U_s( iS, iCF, iX, k ) = SumVar_U( iCF, iX, k );
          } );

      auto StageDataj = Kokkos::subview( StageData, iS, Kokkos::ALL );
      Grid_s[iS].UpdateGrid( StageDataj );

      auto Us_j =
          Kokkos::subview( U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      S_Limiter->ApplySlopeLimiter( Us_j, &Grid_s[iS], Basis );
      ApplyBoundEnforcingLimiter( Us_j, Basis, eos );
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
                                                  { order, ihi + 2, nvars } ),
          KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
            U( iCF, iX, k ) +=
                dt * explicit_tableau_.b_i( iS ) * dUs_j( iCF, iX, k );
          } );

      Kokkos::parallel_for(
          "Timestepper::StageData::final", ihi + 2,
          KOKKOS_LAMBDA( const int iX ) {
            StageData( 0, iX ) +=
                dt * Flux_Uj( iX ) * explicit_tableau_.b_i( iS );
          } );
      auto StageDataj = Kokkos::subview( StageData, 0, Kokkos::ALL );
      Grid_s[iS].UpdateGrid( StageDataj );
    }

    Grid = Grid_s[nStages];
    S_Limiter->ApplySlopeLimiter( U, &Grid, Basis );
    ApplyBoundEnforcingLimiter( U, Basis, eos );
  }

  /**
   * Update radiation solution with SSPRK methods
   **/
  template <typename T>
  void UpdateRadiation( T ComputeIncrementRad, const Real dt, State *state,
                        GridStructure &Grid, const ModalBasis *Basis,
                        const EOS *eos, SlopeLimiter *S_Limiter,
                        const Options opts ) {
    UpdateRadiation_IMEX( ComputeIncrementRad, dt, state, Grid, Basis, eos,
                          S_Limiter, opts );
  }
  /**
   * Update radiation solution with SSPRK methods
   **/
  template <typename T>
  void UpdateRadiation_IMEX( T ComputeIncrementRad, const Real dt, State *state,
                             GridStructure &Grid, const ModalBasis *Basis,
                             const EOS *eos, SlopeLimiter *S_Limiter,
                             const Options opts ) {

    const auto &order = Basis->Get_Order( );
    const auto &ihi   = Grid.Get_ihi( );

    auto uCF = state->Get_uCF( );
    auto uCR = state->Get_uCR( );

    const int nvars = uCR.extent( 0 );

    for ( unsigned short int iS = 0; iS < nStages; iS++ ) {
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper :: Rad :: 1",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { order, ihi + 2, nvars } ),
          KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
            SumVar_U( iCR, iX, k ) = uCR( iCR, iX, k );
          } );

      // --- Inner update loop ---

      for ( int j = 0; j < iS; j++ ) {
        auto Us_j =
            Kokkos::subview( U_s_r, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto dUs_j =
            Kokkos::subview( dU_s_r, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto Flux_Uj = Kokkos::subview( Flux_U, j, Kokkos::ALL );
        ComputeIncrementRad( Us_j, uCF, Grid_s[j], Basis, eos, dUs_j, Flux_q,
                             dFlux_num, uCF_F_L, uCF_F_R, Flux_Uj, Flux_P,
                             opts );

        // inner sum
        Kokkos::parallel_for(
            "Timestepper :: Rad :: 2",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                    { order, ihi + 2, nvars } ),
            KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
              SumVar_U( iCR, iX, k ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * dUs_j( iCR, iX, k );
            } );
      } // End inner loop

      // set U_s
      Kokkos::parallel_for(
          "Timestepper :: Rad :: 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { order, ihi + 2, nvars } ),
          KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
            U_s_r( iS, iCR, iX, k ) = SumVar_U( iCR, iX, k );
          } );

      auto Us_j =
          Kokkos::subview( U_s_r, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      // S_Limiter->ApplySlopeLimiter( Us_j, &Grid, Basis );
      // ApplyBoundEnforcingLimiter( Us_j, Basis, eos );
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
                                                  { order, ihi + 2, nvars } ),
          KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
            uCR( iCR, iX, k ) +=
                dt * explicit_tableau_.b_i( iS ) * dUs_j( iCR, iX, k );
          } );
    }

    // S_Limiter->ApplySlopeLimiter( uCR, &Grid, Basis );
    // ApplyBoundEnforcingLimiter( uCR, Basis, eos );
  }

  /**
   * Update fluid solution with SSPRK methods
   * TODO: register increment funcs with class?
   **/
  template <typename T, typename F, typename G, typename H>
  void UpdateRadHydro( T compute_increment_hydro_explicit,
                       F compute_increment_rad_explicit,
                       G compute_increment_hydro_implicit,
                       H compute_increment_rad_implicit, const Real dt,
                       State *state, GridStructure &Grid,
                       const ModalBasis *Basis, const EOS *eos,
                       SlopeLimiter *S_Limiter, const Options opts ) {

    UpdateRadHydro_IMEX(
        compute_increment_hydro_explicit, compute_increment_rad_explicit,
        compute_increment_hydro_implicit, compute_increment_rad_implicit, dt,
        state, Grid, Basis, eos, S_Limiter, opts );
  }

  /**
   * Explicit fluid update with SSPRK methods
   **/
  template <typename T, typename F, typename G, typename H>
  void UpdateRadHydro_IMEX( T compute_increment_hydro_explicit,
                            F compute_increment_rad_explicit,
                            G compute_increment_hydro_implicit,
                            H compute_increment_rad_implicit, const Real dt,
                            State *state, GridStructure &Grid,
                            const ModalBasis *Basis, const EOS *eos,
                            SlopeLimiter *S_Limiter, const Options opts ) {

    const auto &order = Basis->Get_Order( );
    const auto &ihi   = Grid.Get_ihi( );

    auto uCF = state->Get_uCF( );
    auto uCR = state->Get_uCR( );

    Grid_s[0] = Grid;

    for ( unsigned short int iS = 0; iS < nStages; iS++ ) {
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 0 },
                                                  { order, ihi + 2 } ),
          KOKKOS_LAMBDA( const int k, const int iX ) {
            SumVar_U( 0, iX, k )   = uCF( 0, iX, k );
            SumVar_U( 1, iX, k )   = uCF( 1, iX, k );
            SumVar_U( 2, iX, k )   = uCF( 2, iX, k );
            SumVar_U_r( 0, iX, k ) = uCR( 0, iX, k );
            SumVar_U_r( 1, iX, k ) = uCR( 1, iX, k );
            StageData( iS, iX )    = Grid_s[iS].Get_LeftInterface( iX );
          } );

      // --- Inner update loop ---

      for ( int j = 0; j < iS; j++ ) {
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
                                                    { order, ihi + 2 } ),
            KOKKOS_LAMBDA( const int k, const int iX ) {
              const Real dt_a = dt * explicit_tableau_.a_ij( iS, j );
              SumVar_U( 0, iX, k ) += dt_a * dUs_j_h( 0, iX, k );
              SumVar_U( 1, iX, k ) += dt_a * dUs_j_h( 1, iX, k );
              SumVar_U( 2, iX, k ) += dt_a * dUs_j_h( 2, iX, k );
              SumVar_U_r( 0, iX, k ) += dt_a * dUs_j_r( 0, iX, k );
              SumVar_U_r( 1, iX, k ) += dt_a * dUs_j_r( 1, iX, k );
            } );

        Kokkos::parallel_for(
            "Timestepper::StageData", ihi + 2, KOKKOS_LAMBDA( const int iX ) {
              StageData( iS, iX ) +=
                  dt * explicit_tableau_.a_ij( iS, j ) * Flux_Uj( iX );
            } );
      } // End inner loop

      // set U_s
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, 0 },
                                                  { order, ihi + 2 } ),
          KOKKOS_LAMBDA( const int k, const int iX ) {
            U_s( iS, 0, iX, k )   = SumVar_U( 0, iX, k );
            U_s( iS, 1, iX, k )   = SumVar_U( 1, iX, k );
            U_s( iS, 2, iX, k )   = SumVar_U( 2, iX, k );
            U_s_r( iS, 0, iX, k ) = SumVar_U_r( 0, iX, k );
            U_s_r( iS, 1, iX, k ) = SumVar_U_r( 1, iX, k );
          } );

      auto StageDataj = Kokkos::subview( StageData, iS, Kokkos::ALL );
      Grid_s[iS].UpdateGrid( StageDataj );

      // TODO: slope limit rad
      auto Us_j_h =
          Kokkos::subview( U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
      S_Limiter->ApplySlopeLimiter( Us_j_h, &Grid_s[iS], Basis );
      ApplyBoundEnforcingLimiter( Us_j_h, Basis, eos );
    } // end outer loop

    for ( unsigned short int iS = 0; iS < nStages; iS++ ) {
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
                                                  { order, ihi + 2 } ),
          KOKKOS_LAMBDA( const int k, const int iX ) {
            const Real dt_b = dt * explicit_tableau_.b_i( iS );
            uCF( 0, iX, k ) += dt_b * dUs_i_h( 0, iX, k );
            uCF( 1, iX, k ) += dt_b * dUs_i_h( 1, iX, k );
            uCF( 2, iX, k ) += dt_b * dUs_i_h( 2, iX, k );
            uCR( 0, iX, k ) += dt_b * dUs_i_r( 0, iX, k );
            uCR( 1, iX, k ) += dt_b * dUs_i_r( 1, iX, k );
            // std::printf("dU_i_r %e\n", dUs_i_r( 0, iX, k ));
          } );

      Kokkos::parallel_for(
          "Timestepper::StageData::final", ihi + 2,
          KOKKOS_LAMBDA( const int iX ) {
            StageData( 0, iX ) +=
                dt * Flux_Ui( iX ) * explicit_tableau_.b_i( iS );
          } );
      auto StageDataj = Kokkos::subview( StageData, 0, Kokkos::ALL );
      Grid_s[iS].UpdateGrid( StageDataj );
    }

    // TODO: slope limit rad
    Grid = Grid_s[nStages];
    S_Limiter->ApplySlopeLimiter( uCF, &Grid, Basis );
    ApplyBoundEnforcingLimiter( uCF, Basis, eos );
  }

 private:
  const int mSize;
  const int nStages;
  const int tOrder;
  const std::string BC;

  // tableaus
  // TODO: always have both tableaus?
  // Maybe create an IMEX class... (or new implicit and explicit classes)
  // View2D<Real> a_ij;
  // View2D<Real> b_ij;
  ButcherTableau implicit_tableau_;
  ButcherTableau explicit_tableau_;

  // Hold stage data
  View4D<Real> U_s;
  View4D<Real> dU_s;
  View4D<Real> U_s_r;
  View4D<Real> dU_s_r;
  View3D<Real> SumVar_U;
  View3D<Real> SumVar_U_r;
  std::vector<GridStructure> Grid_s;

  // StageData Holds cell left interface positions
  View2D<Real> StageData;

  // Variables to pass to update step
  View3D<Real> Flux_q;

  View2D<Real> dFlux_num;
  View2D<Real> uCF_F_L;
  View2D<Real> uCF_F_R;
  View2D<Real> Flux_U;
  View1D<Real> Flux_P;
};

#endif // TIMESTEPPER_HPP_
