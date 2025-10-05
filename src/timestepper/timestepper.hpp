/**
 * @file timestepper.hpp
 * --------------
 *
 * @brief Primary time marching routine.
 *
 * @details Timestppers for hydro and rad hydro.
 *          Uses explicit for transport terms and implicit for coupling.
 *
 * TODO(astrobaker) move to calling step<fluid> / step<radhydro>
 */

#pragma once

#include "abstractions.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/hydro_package.hpp"
#include "interface/packages_base.hpp"
#include "limiters/bound_enforcing_limiter.hpp"
#include "limiters/slope_limiter.hpp"
#include "problem_in.hpp"
#include "radiation/radhydro_package.hpp"
#include "state/state.hpp"
#include "timestepper/tableau.hpp"

namespace athelas {

using bc::BoundaryConditions;
using fluid::HydroPackage;
using radiation::RadHydroPackage;

class TimeStepper {

 public:
  // TODO(astrobarker): Is it possible to initialize grid_s_ from grid directly?
  TimeStepper(const ProblemIn *pin, GridStructure *grid, eos::EOS *eos);

  void initialize_timestepper();

  /**
   * Update fluid solution with SSPRK methods
   **/
  void step(PackageManager *pkgs, State *state, GridStructure &grid,
            const double t, const double dt, SlopeLimiter *sl_hydro) {

    // hydro explicit update
    update_fluid_explicit(pkgs, state, grid, t, dt, sl_hydro);
  }

  /**
   * Explicit fluid update with SSPRK methods
   **/
  void update_fluid_explicit(PackageManager *pkgs, State *state,
                             GridStructure &grid, const double t,
                             const double dt, SlopeLimiter *sl_hydro) {

    const auto &order = grid.get_n_nodes();
    const auto &ihi = grid.get_ihi();
    const auto evolve_nickel = state->nickel_evolved();

    auto U = state->u_cf();
    auto U_s = state->u_cf_stages();

    const int nvars = U.extent(2);

    grid_s_[0] = grid;

    TimeStepInfo dt_info{.t = t, .dt = dt, .dt_a = dt, .stage = 0};

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                 {ihi + 2, order, nvars}),
          KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
            SumVar_U_(ix, k, q) = U(ix, k, q);
            stage_data_(iS, ix) = grid.get_left_interface(ix);
          });
      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions = comps->mass_fractions();
        const auto *const species_indexer = comps->species_indexer();
        const auto ind_ni = species_indexer->get<int>("ni56");
        const auto ind_co = species_indexer->get<int>("co56");
        const auto ind_fe = species_indexer->get<int>("fe56");
        Kokkos::parallel_for(
            "Timestepper :: Ni :: Reset SumVar", ihi + 2,
            KOKKOS_CLASS_LAMBDA(const int ix) {
              SumVar_U_(ix, 0, 3 + ind_ni) = mass_fractions(ix, 0, ind_ni);
              SumVar_U_(ix, 0, 3 + ind_co) = mass_fractions(ix, 0, ind_co);
              SumVar_U_(ix, 0, 3 + ind_fe) = mass_fractions(ix, 0, ind_fe);
            });
      }

      // --- Inner update loop ---

      for (int j = 0; j < iS; ++j) {
        dt_info.stage = j;
        dt_info.t = t + integrator_.explicit_tableau.c_i(j);
        auto dUs_j =
            Kokkos::subview(dU_s_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        pkgs->fill_derived(state, grid_s_[j], dt_info);
        pkgs->update_explicit(state, dUs_j, grid_s_[j], dt_info);

        // inner sum
        const double dt_a_ex = dt * integrator_.explicit_tableau.a_ij(iS, j);
        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                   {ihi + 2, order, nvars}),
            KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
              SumVar_U_(ix, k, q) += dt_a_ex * dUs_j(ix, k, q);
            });

        if (evolve_nickel) {
          Kokkos::parallel_for(
              "Timestepper 4",
              Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 3},
                                                     {ihi + 2, nvars + 3}),
              KOKKOS_CLASS_LAMBDA(const int ix, const int q) {
                SumVar_U_(ix, 0, q) += dt_a_ex * dUs_j(ix, 0, q);
              });
        }

        Kokkos::parallel_for(
            "Timestepper::stage_data_", ihi + 2,
            KOKKOS_CLASS_LAMBDA(const int ix) {
              stage_data_(iS, ix) +=
                  dt_a_ex *
                  pkgs->get_package<HydroPackage>("Hydro")->get_flux_u(j, ix);
            });
      } // End inner loop

      // set U_s
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                 {ihi + 2, order, nvars}),
          KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
            U_s(iS, ix, k, q) = SumVar_U_(ix, k, q);
          });

      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions_stages = comps->mass_fractions_stages();
        const auto *const species_indexer = comps->species_indexer();
        const auto ind_ni = species_indexer->get<int>("ni56");
        const auto ind_co = species_indexer->get<int>("co56");
        const auto ind_fe = species_indexer->get<int>("fe56");
        Kokkos::parallel_for(
            "Timestepper :: Nickel :: Set Us", ihi + 2,
            KOKKOS_CLASS_LAMBDA(const int ix) {
              mass_fractions_stages(iS, ix, 0, ind_ni) =
                  SumVar_U_(ix, 0, 3 + ind_ni);
              mass_fractions_stages(iS, ix, 0, ind_co) =
                  SumVar_U_(ix, 0, 3 + ind_co);
              mass_fractions_stages(iS, ix, 0, ind_fe) =
                  SumVar_U_(ix, 0, 3 + ind_fe);
            });
      }

      auto stage_data_j = Kokkos::subview(stage_data_, iS, Kokkos::ALL);
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(stage_data_j);

      auto Us_j =
          Kokkos::subview(U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      apply_slope_limiter(sl_hydro, Us_j, &grid_s_[iS],
                          pkgs->get_package<HydroPackage>("Hydro")->get_basis(),
                          eos_);
      bel::apply_bound_enforcing_limiter(
          Us_j, pkgs->get_package<HydroPackage>("Hydro")->get_basis());
    } // end outer loop

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS);
      auto dUs_j =
          Kokkos::subview(dU_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

      pkgs->fill_derived(state, grid_s_[iS], dt_info);
      pkgs->update_explicit(state, dUs_j, grid_s_[iS], dt_info);

      const double dt_b_ex = dt * integrator_.explicit_tableau.b_i(iS);
      Kokkos::parallel_for(
          "Timestepper :: u^(n+1) from the stages",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                 {ihi + 2, order, nvars}),
          KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
            U(ix, k, q) += dt_b_ex * dUs_j(ix, k, q);
          });
      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions = comps->mass_fractions();
        const auto *const species_indexer = comps->species_indexer();
        const auto ind_ni = species_indexer->get<int>("ni56");
        const auto ind_co = species_indexer->get<int>("co56");
        const auto ind_fe = species_indexer->get<int>("fe56");
        Kokkos::parallel_for(
            "Timestepper :: Nickel :: u^(n+1) from the stages", ihi + 2,
            KOKKOS_CLASS_LAMBDA(const int ix) {
              mass_fractions(ix, 0, ind_ni) +=
                  dt_b_ex * dUs_j(ix, 0, 3 + ind_ni);
              mass_fractions(ix, 0, ind_co) +=
                  dt_b_ex * dUs_j(ix, 0, 3 + ind_co);
              mass_fractions(ix, 0, ind_fe) +=
                  dt_b_ex * dUs_j(ix, 0, 3 + ind_fe);
            });
      }

      Kokkos::parallel_for(
          "Timestepper::stage_data_::final", ihi + 2,
          KOKKOS_CLASS_LAMBDA(const int ix) {
            stage_data_(0, ix) +=
                dt_b_ex *
                pkgs->get_package<HydroPackage>("Hydro")->get_flux_u(iS, ix);
          });
      auto stage_data_j = Kokkos::subview(stage_data_, 0, Kokkos::ALL);
      grid_s_[iS] = grid;
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
  void step_imex(PackageManager *pkgs, State *state, GridStructure &grid,
                 const double t, const double dt, SlopeLimiter *sl_hydro,
                 SlopeLimiter *sl_rad) {

    update_rad_hydro_imex(pkgs, state, grid, t, dt, sl_hydro, sl_rad);
  }

  /**
   * Fully coupled IMEX rad hydro update with SSPRK methods
   **/
  void update_rad_hydro_imex(PackageManager *pkgs, State *state,
                             GridStructure &grid, const double t,
                             const double dt, SlopeLimiter *sl_hydro,
                             SlopeLimiter *sl_rad) {

    const auto &order = grid.get_n_nodes();
    const auto &ihi = grid.get_ihi();
    const auto evolve_nickel = state->nickel_evolved();

    auto uCF = state->u_cf();
    auto U_s = state->u_cf_stages();

    const int nvars = uCF.extent(2);

    grid_s_[0] = grid;

    // TODO(astrobarker) pass in time
    TimeStepInfo dt_info{.t = t, .dt = dt, .dt_a = dt, .stage = 0};

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS);
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                 {ihi + 2, order, nvars}),
          KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
            SumVar_U_(ix, k, q) = uCF(ix, k, q);
            stage_data_(iS, ix) = grid_s_[iS].get_left_interface(ix);
          });
      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions = comps->mass_fractions();
        const auto *const species_indexer = comps->species_indexer();
        const auto ind_ni = species_indexer->get<int>("ni56");
        const auto ind_co = species_indexer->get<int>("co56");
        const auto ind_fe = species_indexer->get<int>("fe56");
        Kokkos::parallel_for(
            "Timestepper :: Ni :: Reset SumVar", ihi + 2,
            KOKKOS_CLASS_LAMBDA(const int ix) {
              SumVar_U_(ix, 0, 5 + ind_ni) = mass_fractions(ix, 0, ind_ni);
              SumVar_U_(ix, 0, 5 + ind_co) = mass_fractions(ix, 0, ind_co);
              SumVar_U_(ix, 0, 5 + ind_fe) = mass_fractions(ix, 0, ind_fe);
            });
      }

      // --- Inner update loop ---

      for (int j = 0; j < iS; ++j) {
        dt_info.stage = j;
        dt_info.t = t + integrator_.explicit_tableau.c_i(j);
        const double dt_a = dt * integrator_.explicit_tableau.a_ij(iS, j);
        const double dt_a_im = dt * integrator_.implicit_tableau.a_ij(iS, j);
        auto dUs_j =
            Kokkos::subview(dU_s_, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

        pkgs->fill_derived(state, grid_s_[j], dt_info);
        pkgs->update_explicit(state, dUs_j, grid_s_[j], dt_info);

        // inner sum
        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                   {ihi + 2, order, nvars}),
            KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
              SumVar_U_(ix, k, q) += dt_a * dUs_j(ix, k, q);
            });

        if (evolve_nickel) {
          Kokkos::parallel_for(
              "Timestepper 4",
              Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 5},
                                                     {ihi + 2, nvars + 5}),
              KOKKOS_CLASS_LAMBDA(const int ix, const int q) {
                SumVar_U_(ix, 0, q) += dt_a * dUs_j(ix, 0, q);
              });
        }

        pkgs->update_implicit(state, dUs_j, grid_s_[j], dt_info);

        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 1},
                                                   {ihi + 2, order, nvars}),
            KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
              SumVar_U_(ix, k, q) += dt_a_im * dUs_j(ix, k, q);
            });

        Kokkos::parallel_for(
            "Timestepper::stage_data_", ihi + 2,
            KOKKOS_CLASS_LAMBDA(const int ix) {
              stage_data_(iS, ix) +=
                  dt_a * pkgs->get_package<RadHydroPackage>("RadHydro")
                             ->get_flux_u(j, ix);
            });
      } // End inner loop

      auto stage_data_j = Kokkos::subview(stage_data_, iS, Kokkos::ALL);
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(stage_data_j);

      // set U_s
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                 {ihi + 2, order, nvars}),
          KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
            U_s(iS, ix, k, q) = SumVar_U_(ix, k, q);
          });

      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions_stages = comps->mass_fractions_stages();
        const auto *const species_indexer = comps->species_indexer();
        const auto ind_ni = species_indexer->get<int>("ni56");
        const auto ind_co = species_indexer->get<int>("co56");
        const auto ind_fe = species_indexer->get<int>("fe56");
        Kokkos::parallel_for(
            "Timestepper :: Nickel :: Set Us", ihi + 2,
            KOKKOS_CLASS_LAMBDA(const int ix) {
              mass_fractions_stages(iS, ix, 0, ind_ni) =
                  SumVar_U_(ix, 0, 5 + ind_ni);
              mass_fractions_stages(iS, ix, 0, ind_co) =
                  SumVar_U_(ix, 0, 5 + ind_co);
              mass_fractions_stages(iS, ix, 0, ind_fe) =
                  SumVar_U_(ix, 0, 5 + ind_fe);
            });
      }

      // NOTE: The limiting strategies in this function will fail if
      // the pkg does not have access to a rad_basis and fluid_basis
      auto Us_j =
          Kokkos::subview(U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
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
          pkgs->get_package<RadHydroPackage>("RadHydro")->get_fluid_basis());

      // implicit update
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS);
      dt_info.dt_a = dt * integrator_.implicit_tableau.a_ij(iS, iS);
      pkgs->fill_derived(state, grid_s_[iS], dt_info);
      pkgs->update_implicit_iterative(state, SumVar_U_, grid_s_[iS], dt_info);
      pkgs->fill_derived(state, grid_s_[iS], dt_info);

      // set U_s after iterative solve
      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 1},
                                                 {ihi + 2, order, nvars}),
          KOKKOS_CLASS_LAMBDA(const int ix, const int k, const int q) {
            U_s(iS, ix, k, q) = Us_j(ix, k, q);
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
      dt_info.stage = iS;
      const double dt_b = dt * integrator_.explicit_tableau.b_i(iS);
      const double dt_b_im = dt * integrator_.implicit_tableau.b_i(iS);
      auto dUs_ex_i =
          Kokkos::subview(dU_s_, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto dUs_im_i = Kokkos::subview(dU_s_implicit_, iS, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL);

      pkgs->fill_derived(state, grid_s_[iS], dt_info);
      pkgs->update_explicit(state, dUs_ex_i, grid_s_[iS], dt_info);
      pkgs->update_implicit(state, dUs_im_i, grid_s_[iS], dt_info);
      Kokkos::parallel_for(
          "Timestepper :: u^(n+1) from the stages",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ihi + 2, order}),
          KOKKOS_CLASS_LAMBDA(const int ix, const int k) {
            uCF(ix, k, 0) = std::fma(dt_b, dUs_ex_i(ix, k, 0), uCF(ix, k, 0));
            for (int q = 1; q < nvars; ++q) {
              uCF(ix, k, q) = std::fma(dt_b, dUs_ex_i(ix, k, q), uCF(ix, k, q));

              uCF(ix, k, q) =
                  std::fma(dt_b_im, dUs_im_i(ix, k, q), uCF(ix, k, q));
            }
          });
      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions = comps->mass_fractions();
        const auto *const species_indexer = comps->species_indexer();
        const auto ind_ni = species_indexer->get<int>("ni56");
        const auto ind_co = species_indexer->get<int>("co56");
        const auto ind_fe = species_indexer->get<int>("fe56");
        Kokkos::parallel_for(
            "Timestepper :: Nickel :: u^(n+1) from the stages", ihi + 2,
            KOKKOS_CLASS_LAMBDA(const int ix) {
              mass_fractions(ix, 0, ind_ni) +=
                  dt_b * dUs_ex_i(ix, 0, 5 + ind_ni);
              mass_fractions(ix, 0, ind_co) +=
                  dt_b * dUs_ex_i(ix, 0, 5 + ind_co);
              mass_fractions(ix, 0, ind_fe) +=
                  dt_b * dUs_ex_i(ix, 0, 5 + ind_fe);
            });
      }

      Kokkos::parallel_for(
          "Timestepper::stage_data_::final", ihi + 2,
          KOKKOS_CLASS_LAMBDA(const int ix) {
            stage_data_(0, ix) +=
                dt_b * pkgs->get_package<RadHydroPackage>("RadHydro")
                           ->get_flux_u(iS, ix);
          });
      auto stage_data_j = Kokkos::subview(stage_data_, 0, Kokkos::ALL); // HERE
      grid_s_[iS] = grid;
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

  [[nodiscard]] auto n_stages() const noexcept -> int;
  [[nodiscard]] static auto nvars_evolved(const ProblemIn *pin) noexcept -> int;

 private:
  int nvars_evolved_;
  int mSize_;

  // tableaus
  RKIntegrator integrator_;

  int nStages_;
  int tOrder_;

  // Hold stage data
  View4D<double> dU_s_;
  View4D<double> dU_s_implicit_;
  View3D<double> SumVar_U_;
  std::vector<GridStructure> grid_s_;

  // stage_data_ Holds cell left interface positions
  View2D<double> stage_data_;

  // Variables to pass to update step

  // hold eos::EOS ptr for convenience
  eos::EOS *eos_;
};

} // namespace athelas
