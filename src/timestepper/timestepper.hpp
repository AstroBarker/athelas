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

#include "bc/boundary_conditions_base.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/hydro_package.hpp"
#include "interface/packages_base.hpp"
#include "kokkos_abstraction.hpp"
#include "limiters/bound_enforcing_limiter.hpp"
#include "limiters/slope_limiter.hpp"
#include "loop_layout.hpp"
#include "problem_in.hpp"
#include "radiation/radhydro_package.hpp"
#include "state/state.hpp"
#include "timestepper/tableau.hpp"
#include "utils/abstractions.hpp"

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
    const auto evolve_nickel = state->nickel_evolved();

    auto U = state->u_cf();
    auto U_s = state->u_cf_stages();

    const int nvars = U.extent(2);
    static const IndexRange ib(grid.domain<Domain::Entire>());
    static const IndexRange kb(order);
    static const IndexRange vb(nvars);

    grid_s_[0] = grid;

    TimeStepInfo dt_info{.t = t, .dt = dt, .dt_a = dt, .stage = 0};

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      // re-zero the summation variables `SumVar`
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: Reset sumvar",
          DevExecSpace(), ib.s, ib.e, kb.s, kb.e, vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
            SumVar_U_(i, k, v) = U(i, k, v);
            stage_data_(iS, i) = grid.get_left_interface(i);
          });
      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions = comps->mass_fractions();
        const auto *const species_indexer = comps->species_indexer();
        static const auto ind_ni = species_indexer->get<int>("ni56");
        static const auto ind_co = species_indexer->get<int>("co56");
        static const auto ind_fe = species_indexer->get<int>("fe56");
        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN,
            "Timestepper :: EX :: Ni :: Reset sumvar", DevExecSpace(), ib.s,
            ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              SumVar_U_(i, 0, 3 + ind_ni) = mass_fractions(i, 0, ind_ni);
              SumVar_U_(i, 0, 3 + ind_co) = mass_fractions(i, 0, ind_co);
              SumVar_U_(i, 0, 3 + ind_fe) = mass_fractions(i, 0, ind_fe);
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
        athelas::par_for(
            DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: update sumvar",
            DevExecSpace(), ib.s, ib.e, kb.s, kb.e, vb.s, vb.e,
            KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
              SumVar_U_(i, k, v) += dt_a_ex * dUs_j(i, k, v);
            });

        if (evolve_nickel) {
          // this updates mass fractions
          athelas::par_for(
              DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: Ni :: update sumvar",
              DevExecSpace(), ib.s, ib.e, 3, nvars + 2,
              KOKKOS_CLASS_LAMBDA(const int i, const int v) {
                SumVar_U_(i, 0, v) += dt_a_ex * dUs_j(i, 0, v);
              });
        }

        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: EX :: grid",
            DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              stage_data_(iS, i) +=
                  dt_a_ex *
                  pkgs->get_package<HydroPackage>("Hydro")->get_flux_u(j, i);
            });
      } // End inner loop

      // set U_s
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: Set Us", DevExecSpace(),
          ib.s, ib.e, kb.s, kb.e, vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
            U_s(iS, i, k, v) = SumVar_U_(i, k, v);
          });

      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions_stages = comps->mass_fractions_stages();
        const auto *const species_indexer = comps->species_indexer();
        static const auto ind_ni = species_indexer->get<int>("ni56");
        static const auto ind_co = species_indexer->get<int>("co56");
        static const auto ind_fe = species_indexer->get<int>("fe56");
        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: EX :: Ni :: Set Us",
            DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              mass_fractions_stages(iS, i, 0, ind_ni) =
                  SumVar_U_(i, 0, 3 + ind_ni);
              mass_fractions_stages(iS, i, 0, ind_co) =
                  SumVar_U_(i, 0, 3 + ind_co);
              mass_fractions_stages(iS, i, 0, ind_fe) =
                  SumVar_U_(i, 0, 3 + ind_fe);
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
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: Finalize", DevExecSpace(),
          ib.s, ib.e, kb.s, kb.e, vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
            U(i, k, v) += dt_b_ex * dUs_j(i, k, v);
          });

      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions = comps->mass_fractions();
        const auto *const species_indexer = comps->species_indexer();
        static const auto ind_ni = species_indexer->get<int>("ni56");
        static const auto ind_co = species_indexer->get<int>("co56");
        static const auto ind_fe = species_indexer->get<int>("fe56");
        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: EX :: Ni :: Finalize",
            DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              mass_fractions(i, 0, ind_ni) += dt_b_ex * dUs_j(i, 0, 3 + ind_ni);
              mass_fractions(i, 0, ind_co) += dt_b_ex * dUs_j(i, 0, 3 + ind_co);
              mass_fractions(i, 0, ind_fe) += dt_b_ex * dUs_j(i, 0, 3 + ind_fe);
            });
      }

      athelas::par_for(
          DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: EX :: Finalize grid",
          DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
            stage_data_(0, i) +=
                dt_b_ex *
                pkgs->get_package<HydroPackage>("Hydro")->get_flux_u(iS, i);
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
    const auto evolve_nickel = state->nickel_evolved();

    auto uCF = state->u_cf();
    auto U_s = state->u_cf_stages();

    const int nvars = uCF.extent(2);
    static const IndexRange ib(grid.domain<Domain::Entire>());
    static const IndexRange kb(order);
    static const IndexRange vb(nvars);

    grid_s_[0] = grid;

    // TODO(astrobarker) pass in time
    TimeStepInfo dt_info{.t = t, .dt = dt, .dt_a = dt, .stage = 0};

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS);
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Reset sumvar",
          DevExecSpace(), ib.s, ib.e, kb.s, kb.e, vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
            SumVar_U_(i, k, v) = uCF(i, k, v);
            stage_data_(iS, i) = grid_s_[iS].get_left_interface(i);
          });

      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions = comps->mass_fractions();
        const auto *const species_indexer = comps->species_indexer();
        static const auto ind_ni = species_indexer->get<int>("ni56");
        static const auto ind_co = species_indexer->get<int>("co56");
        static const auto ind_fe = species_indexer->get<int>("fe56");
        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN,
            "Timestepper :: IMEX :: Ni :: Reset sumvar", DevExecSpace(), ib.s,
            ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              SumVar_U_(i, 0, 5 + ind_ni) = mass_fractions(i, 0, ind_ni);
              SumVar_U_(i, 0, 5 + ind_co) = mass_fractions(i, 0, ind_co);
              SumVar_U_(i, 0, 5 + ind_fe) = mass_fractions(i, 0, ind_fe);
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
        athelas::par_for(
            DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Update sumvar",
            DevExecSpace(), ib.s, ib.e, kb.s, kb.e, vb.s, vb.e,
            KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
              SumVar_U_(i, k, v) += dt_a * dUs_j(i, k, v);
            });

        if (evolve_nickel) {
          // NOTE: Is this indexing going to be correct in more general
          // settings?
          athelas::par_for(
              DEFAULT_LOOP_PATTERN,
              "Timestepper :: IMEX :: Ni :: Update sumvar", DevExecSpace(),
              ib.s, ib.e, 5, nvars + 5,
              KOKKOS_CLASS_LAMBDA(const int ix, const int q) {
                SumVar_U_(ix, 0, q) += dt_a * dUs_j(ix, 0, q);
              });
        }

        pkgs->update_implicit(state, dUs_j, grid_s_[j], dt_info);

        athelas::par_for(
            DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Implicit delta",
            DevExecSpace(), ib.s, ib.e, kb.s, kb.e, 1, vb.e,
            KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
              SumVar_U_(i, k, v) += dt_a_im * dUs_j(i, k, v);
            });

        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: IMEX :: Update grid",
            DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              stage_data_(iS, i) +=
                  dt_a * pkgs->get_package<RadHydroPackage>("RadHydro")
                             ->get_flux_u(j, i);
            });
      } // End inner loop

      auto stage_data_j = Kokkos::subview(stage_data_, iS, Kokkos::ALL);
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(stage_data_j);

      // set U_s
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Update Us",
          DevExecSpace(), ib.s, ib.e, kb.s, kb.e, vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
            U_s(iS, i, k, v) = SumVar_U_(i, k, v);
          });

      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions_stages = comps->mass_fractions_stages();
        const auto *const species_indexer = comps->species_indexer();
        static const auto ind_ni = species_indexer->get<int>("ni56");
        static const auto ind_co = species_indexer->get<int>("co56");
        static const auto ind_fe = species_indexer->get<int>("fe56");
        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: IMEX :: Ni :: Set Us",
            DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              mass_fractions_stages(iS, i, 0, ind_ni) =
                  SumVar_U_(i, 0, 5 + ind_ni);
              mass_fractions_stages(iS, i, 0, ind_co) =
                  SumVar_U_(i, 0, 5 + ind_co);
              mass_fractions_stages(iS, i, 0, ind_fe) =
                  SumVar_U_(i, 0, 5 + ind_fe);
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
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Set Us implicit",
          DevExecSpace(), ib.s, ib.e, kb.s, kb.e, 1, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
            U_s(iS, i, k, v) = Us_j(i, k, v);
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
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Finalize",
          DevExecSpace(), ib.s, ib.e, kb.s, kb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k) {
            uCF(i, k, 0) = std::fma(dt_b, dUs_ex_i(i, k, 0), uCF(i, k, 0));
            for (int v = 1; v < nvars; ++v) {
              uCF(i, k, v) = std::fma(dt_b, dUs_ex_i(i, k, v), uCF(i, k, v));

              uCF(i, k, v) = std::fma(dt_b_im, dUs_im_i(i, k, v), uCF(i, k, v));
            }
          });
      if (evolve_nickel) {
        auto *comps = state->comps();
        auto mass_fractions = comps->mass_fractions();
        const auto *const species_indexer = comps->species_indexer();
        static const auto ind_ni = species_indexer->get<int>("ni56");
        static const auto ind_co = species_indexer->get<int>("co56");
        static const auto ind_fe = species_indexer->get<int>("fe56");
        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN, "Timestepper  :: IMEX :: Ni :: Finalize",
            DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              mass_fractions(i, 0, ind_ni) += dt_b * dUs_ex_i(i, 0, 5 + ind_ni);
              mass_fractions(i, 0, ind_co) += dt_b * dUs_ex_i(i, 0, 5 + ind_co);
              mass_fractions(i, 0, ind_fe) += dt_b * dUs_ex_i(i, 0, 5 + ind_fe);
            });
      }

      athelas::par_for(
          DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: IMEX :: Finalize grid",
          DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
            stage_data_(0, i) +=
                dt_b * pkgs->get_package<RadHydroPackage>("RadHydro")
                           ->get_flux_u(iS, i);
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
