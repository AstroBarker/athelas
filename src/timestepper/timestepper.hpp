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
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/hydro_package.hpp"
#include "limiters/bound_enforcing_limiter.hpp"
#include "limiters/slope_limiter.hpp"
#include "opacity/opac_variant.hpp"
#include "packages/packages_base.hpp"
#include "problem_in.hpp"
#include "radiation/radhydro_package.hpp"
#include "solvers/root_finders.hpp"
#include "state/state.hpp"
#include "timestepper/tableau.hpp"

using bc::BoundaryConditions;
using fluid::HydroPackage;
using radiation::RadHydroPackage;

class TimeStepper {

 public:
  // TODO(astrobarker): Is it possible to initialize grid_s_ from grid directly?
  TimeStepper(const ProblemIn* pin, GridStructure* grid, EOS* eos);

  void initialize_timestepper();

  /**
   * Update fluid solution with SSPRK methods
   **/
  void step(PackageManager* pkgs, State* state, GridStructure& grid,
            const double dt, SlopeLimiter* sl_hydro, const Options* opts) {

    // hydro explicit update
    update_fluid_explicit(pkgs, state, grid, dt, sl_hydro, opts);
  }

  /**
   * Explicit fluid update with SSPRK methods
   **/
  void update_fluid_explicit(PackageManager* pkgs, State* state,
                             GridStructure& grid, const double dt,
                             SlopeLimiter* sl_hydro, const Options* opts) {

    const auto& order = opts->max_order;
    const auto& ihi   = grid.get_ihi();

    auto U = state->get_u_cf();

    const int nvars = U.extent(0);

    grid_s_[0] = grid;

    TimeStepInfo dt_info{.t = 0.0, .dt = dt, .dt_a = dt, .stage = 0};

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                 {nvars, ihi + 2, order}),
          KOKKOS_CLASS_LAMBDA(const int iCF, const int iX, const int k) {
            SumVar_U_(iCF, iX, k) = U(iCF, iX, k);
            stage_data_(iS, iX)   = grid.get_left_interface(iX);
            // stage_data_( iS, iX )    = grid_s_[iS].get_left_interface( iX );
          });

      // --- Inner update loop ---

      for (int j = 0; j < iS; ++j) {
        dt_info.stage = j;
        auto Us_j =
            Kokkos::subview(U_s_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        auto dUs_j =
            Kokkos::subview(dU_s_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        auto flux_u_j = Kokkos::subview(flux_u_, j, Kokkos::ALL);
        pkgs->update_explicit(Us_j, dUs_j, grid_s_[j], dt_info);

        // inner sum
        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                   {nvars, ihi + 2, order}),
            KOKKOS_CLASS_LAMBDA(const int iCF, const int iX, const int k) {
              SumVar_U_(iCF, iX, k) +=
                  dt * integrator_.explicit_tableau.a_ij(iS, j) *
                  dUs_j(iCF, iX, k);
            });

        Kokkos::parallel_for(
            "Timestepper::stage_data_", ihi + 2,
            KOKKOS_CLASS_LAMBDA(const int iX) {
              stage_data_(iS, iX) +=
                  dt * integrator_.explicit_tableau.a_ij(iS, j) *
                  pkgs->get_package<HydroPackage>("Hydro")->get_flux_u(j, iX);
            });
      } // End inner loop

      // set U_s
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                 {nvars, ihi + 2, order}),
          KOKKOS_CLASS_LAMBDA(const int iCF, const int iX, const int k) {
            U_s_(iS, iCF, iX, k) = SumVar_U_(iCF, iX, k);
          });

      auto stage_data_j = Kokkos::subview(stage_data_, iS, Kokkos::ALL);
      grid_s_[iS].update_grid(stage_data_j);

      auto Us_j =
          Kokkos::subview(U_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      apply_slope_limiter(sl_hydro, SumVar_U_, &grid_s_[iS],
                          pkgs->get_package<HydroPackage>("Hydro")->get_basis(),
                          eos_);
      bel::apply_bound_enforcing_limiter(
          Us_j, pkgs->get_package<HydroPackage>("Hydro")->get_basis());
    } // end outer loop

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      auto Us_j =
          Kokkos::subview(U_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto dUs_j =
          Kokkos::subview(dU_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto flux_u_j = Kokkos::subview(flux_u_, iS, Kokkos::ALL);

      pkgs->update_explicit(Us_j, dUs_j, grid_s_[iS], dt_info);
      Kokkos::parallel_for(
          "Timestepper :: u^(n+1) from the stages",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                 {nvars, ihi + 2, order}),
          KOKKOS_CLASS_LAMBDA(const int iCF, const int iX, const int k) {
            U(iCF, iX, k) +=
                dt * integrator_.explicit_tableau.b_i(iS) * dUs_j(iCF, iX, k);
          });

      Kokkos::parallel_for(
          "Timestepper::stage_data_::final", ihi + 2,
          KOKKOS_CLASS_LAMBDA(const int iX) {
            stage_data_(0, iX) +=
                dt *
                pkgs->get_package<HydroPackage>("Hydro")->get_flux_u(iS, iX) *
                integrator_.explicit_tableau.b_i(iS);
          });
      auto stage_data_j = Kokkos::subview(stage_data_, 0, Kokkos::ALL);
      grid_s_[iS].update_grid(stage_data_j);
    }

    grid = grid_s_[nStages_ - 1];
    apply_slope_limiter(sl_hydro, U, &grid,
                        pkgs->get_package<HydroPackage>("Hydro")->get_basis(),
                        eos_);
    bel::apply_bound_enforcing_limiter(
        U, pkgs->get_package<HydroPackage>("Hydro")->get_basis());
  }

  /**
   * Update rad hydro solution with SSPRK methods
   **/
  void step_imex(PackageManager* pkgs, State* state, GridStructure& grid,
                 const double dt, SlopeLimiter* sl_hydro, SlopeLimiter* sl_rad,
                 const Options* opts) {

    update_rad_hydro_imex(pkgs, state, grid, dt, sl_hydro, sl_rad, opts);
  }

  /**
   * Fully coupled IMEX rad hydro update with SSPRK methods
   **/
  void update_rad_hydro_imex(PackageManager* pkgs, State* state,
                             GridStructure& grid, const double dt,
                             SlopeLimiter* sl_hydro, SlopeLimiter* sl_rad,
                             const Options* opts) {

    const auto& order = opts->max_order;
    const auto& ihi   = grid.get_ihi();

    auto uCF = state->get_u_cf();

    grid_s_[0] = grid;

    // TODO(astrobarker) pass in time
    TimeStepInfo dt_info{.t = 0.0, .dt = dt, .dt_a = dt, .stage = 0};

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ihi + 2, order}),
          KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
            SumVar_U_(0, iX, k) = uCF(0, iX, k);
            SumVar_U_(1, iX, k) = uCF(1, iX, k);
            SumVar_U_(2, iX, k) = uCF(2, iX, k);
            SumVar_U_(3, iX, k) = uCF(3, iX, k);
            SumVar_U_(4, iX, k) = uCF(4, iX, k);
            stage_data_(iS, iX) = grid_s_[iS].get_left_interface(iX);
          });

      // --- Inner update loop ---

      for (int j = 0; j < iS; ++j) {
        dt_info.stage        = j;
        const double dt_a    = dt * integrator_.explicit_tableau.a_ij(iS, j);
        const double dt_a_im = dt * integrator_.implicit_tableau.a_ij(iS, j);
        auto Us_j =
            Kokkos::subview(U_s_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        auto dUs_j =
            Kokkos::subview(dU_s_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        auto flux_u_j = Kokkos::subview(flux_u_, j, Kokkos::ALL);

        pkgs->update_explicit(Us_j, dUs_j, grid_s_[j], dt_info);

        // inner sum
        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ihi + 2, order}),
            KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
              SumVar_U_(0, iX, k) += dt_a * dUs_j(0, iX, k);
              SumVar_U_(1, iX, k) += dt_a * dUs_j(1, iX, k);
              SumVar_U_(2, iX, k) += dt_a * dUs_j(2, iX, k);
              SumVar_U_(3, iX, k) += dt_a * dUs_j(3, iX, k);
              SumVar_U_(4, iX, k) += dt_a * dUs_j(4, iX, k);
            });

        pkgs->update_implicit(Us_j, dUs_j, grid_s_[j], dt_info);

        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ihi + 2, order}),
            KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
              SumVar_U_(1, iX, k) += dt_a_im * dUs_j(1, iX, k);
              SumVar_U_(2, iX, k) += dt_a_im * dUs_j(2, iX, k);
              SumVar_U_(3, iX, k) += dt_a_im * dUs_j(3, iX, k);
              SumVar_U_(4, iX, k) += dt_a_im * dUs_j(4, iX, k);
            });

        Kokkos::parallel_for(
            "Timestepper::stage_data_", ihi + 2,
            KOKKOS_CLASS_LAMBDA(const int iX) {
              stage_data_(j, iX) +=
                  dt * integrator_.explicit_tableau.a_ij(iS, j) * flux_u_j(iX);
            });
      } // End inner loop

      auto stage_data_j = Kokkos::subview(stage_data_, iS, Kokkos::ALL);
      grid_s_[iS].update_grid(stage_data_j);

      // set U_s
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ihi + 2, order}),
          KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
            U_s_(iS, 0, iX, k) = SumVar_U_(0, iX, k);
            U_s_(iS, 1, iX, k) = SumVar_U_(1, iX, k);
            U_s_(iS, 2, iX, k) = SumVar_U_(2, iX, k);
            U_s_(iS, 3, iX, k) = SumVar_U_(3, iX, k);
            U_s_(iS, 4, iX, k) = SumVar_U_(4, iX, k);
          });

      // NOTE: The limiting strategies in this function will fail if
      // the pkg does not have access to a rad_basis and fluid_basis
      auto Us_j =
          Kokkos::subview(U_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      // limiting madness
      apply_slope_limiter(
          sl_hydro, Us_j, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_fluid_basis(),
          eos_);
      apply_slope_limiter(
          sl_rad, Us_j, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_rad_basis(),
          eos_);
      apply_slope_limiter(
          sl_rad, SumVar_U_, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_rad_basis(),
          eos_);
      apply_slope_limiter(
          sl_hydro, SumVar_U_, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_fluid_basis(),
          eos_);
      bel::apply_bound_enforcing_limiter(
          Us_j,
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_fluid_basis());
      bel::apply_bound_enforcing_limiter_rad(
          Us_j,
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_rad_basis());
      bel::apply_bound_enforcing_limiter_rad(
          SumVar_U_,
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_rad_basis());
      bel::apply_bound_enforcing_limiter(
          SumVar_U_,
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_rad_basis());

      // implicit update
      dt_info.stage = iS;
      dt_info.dt_a  = dt * integrator_.implicit_tableau.a_ij(iS, iS);
      pkgs->update_implicit_iterative(Us_j, SumVar_U_, grid_s_[iS], dt_info);

      // set U_s after iterative solve
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ihi + 2, order}),
          KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
            U_s_(iS, 1, iX, k) = Us_j(1, iX, k);
            U_s_(iS, 2, iX, k) = Us_j(2, iX, k);
            U_s_(iS, 3, iX, k) = Us_j(3, iX, k);
            U_s_(iS, 4, iX, k) = Us_j(4, iX, k);
          });

      // TODO(astrobarker): slope limit rad
      apply_slope_limiter(
          sl_hydro, Us_j, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_fluid_basis(),
          eos_);
      apply_slope_limiter(
          sl_rad, Us_j, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_rad_basis(),
          eos_);
      bel::apply_bound_enforcing_limiter(
          Us_j,
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_fluid_basis());
      bel::apply_bound_enforcing_limiter_rad(
          Us_j,
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_rad_basis());
    } // end outer loop

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage        = iS;
      const double dt_b    = dt * integrator_.explicit_tableau.b_i(iS);
      const double dt_b_im = dt * integrator_.implicit_tableau.b_i(iS);
      auto Us_i =
          Kokkos::subview(U_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto dUs_ex_i =
          Kokkos::subview(dU_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto dUs_im_i = Kokkos::subview(dU_s_implicit_, iS, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL);
      auto flux_u_i = Kokkos::subview(flux_u_, iS, Kokkos::ALL);

      pkgs->update_explicit(Us_i, dUs_ex_i, grid_s_[iS], dt_info);
      pkgs->update_implicit(Us_i, dUs_im_i, grid_s_[iS], dt_info);
      Kokkos::parallel_for(
          "Timestepper :: u^(n+1) from the stages",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ihi + 2, order}),
          KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
            uCF(0, iX, k) += dt_b * dUs_ex_i(0, iX, k);
            uCF(1, iX, k) += dt_b * dUs_ex_i(1, iX, k);
            uCF(2, iX, k) += dt_b * dUs_ex_i(2, iX, k);
            uCF(3, iX, k) += dt_b * dUs_ex_i(3, iX, k);
            uCF(4, iX, k) += dt_b * dUs_ex_i(4, iX, k);

            uCF(1, iX, k) += dt_b_im * dUs_im_i(1, iX, k);
            uCF(2, iX, k) += dt_b_im * dUs_im_i(2, iX, k);
            uCF(3, iX, k) += dt_b_im * dUs_im_i(3, iX, k);
            uCF(4, iX, k) += dt_b_im * dUs_im_i(4, iX, k);
          });

      Kokkos::parallel_for(
          "Timestepper::stage_data_::final", ihi + 2,
          KOKKOS_CLASS_LAMBDA(const int iX) {
            stage_data_(iS, iX) += dt_b * flux_u_i(iX);
          });
      auto stage_data_j = Kokkos::subview(stage_data_, iS, Kokkos::ALL);
      grid_s_[iS].update_grid(stage_data_j);
    }

    // TODO(astrobarker): slope limit rad
    grid = grid_s_[nStages_ - 1];
    apply_slope_limiter(
        sl_hydro, uCF, &grid,
        pkgs->get_package<RadHydroPackage>("RadHydro")->get_fluid_basis(),
        eos_);
    apply_slope_limiter(
        sl_rad, uCF, &grid,
        pkgs->get_package<RadHydroPackage>("RadHydro")->get_rad_basis(), eos_);
    bel::apply_bound_enforcing_limiter(
        uCF, pkgs->get_package<RadHydroPackage>("RadHydro")->get_fluid_basis());
    bel::apply_bound_enforcing_limiter_rad(
        uCF, pkgs->get_package<RadHydroPackage>("RadHydro")->get_rad_basis());
  }

  [[nodiscard]] auto get_n_stages() const noexcept -> int;

 private:
  int mSize_;

  // tableaus
  RKIntegrator integrator_;

  int nStages_;
  int tOrder_;

  // Hold stage data
  View4D<double> U_s_;
  View4D<double> dU_s_;
  View4D<double> dU_s_implicit_;
  View3D<double> SumVar_U_;
  std::vector<GridStructure> grid_s_;

  // stage_data_ Holds cell left interface positions
  View2D<double> stage_data_;

  // Variables to pass to update step

  View2D<double> flux_u_;

  // hold EOS ptr for convenience
  EOS* eos_;
};
