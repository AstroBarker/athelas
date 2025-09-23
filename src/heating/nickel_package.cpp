#include "heating/nickel_package.hpp"
#include "basis/polynomial_basis.hpp"
#include "constants.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "utils/abstractions.hpp"
#include "utils/utilities.hpp"

namespace ni {
using utilities::to_lower;

NiHeatingPackage::NiHeatingPackage(const ProblemIn *pin, ModalBasis *basis,
                                   const double cfl, const bool active)
    : active_(active), basis_(basis), cfl_(cfl) {
  // set up heating deposition model
  const auto model_str =
      to_lower(pin->param()->get<std::string>("heating.nickel.model"));
  model_ = parse_model(model_str);
}

KOKKOS_FUNCTION
void NiHeatingPackage::update_explicit(const State *const state,
                                       View3D<double> dU,
                                       const GridStructure &grid,
                                       const TimeStepInfo &dt_info) const {
  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  auto *comps = state->comps();

  if (model_ == NiHeatingModel::Schwartz) {
    ni_update<NiHeatingModel::Schwartz>(ucf, comps, dU, grid, dt_info);
  } else [[unlikely]] {
    ni_update<NiHeatingModel::FullTrapping>(ucf, comps, dU, grid, dt_info);
  }
}

KOKKOS_FUNCTION
template <NiHeatingModel Model>
void NiHeatingPackage::ni_update(const View3D<double> ucf,
                                 CompositionData *comps, View3D<double> dU,
                                 const GridStructure &grid,
                                 const TimeStepInfo &dt_info) const {
  const int &nNodes = grid.get_n_nodes();
  const int &order = basis_->get_order();
  static constexpr int ilo = 1;
  static const int &ihi = grid.get_ihi();
  const double time = dt_info.t;

  const auto mass_fractions = comps->mass_fractions();
  const auto charges = comps->charge();
  const auto neutrons = comps->neutron_number();
  const auto *const species_indexer = comps->species_indexer();

  // index gymnastics. dU holds updates for all quantities including
  // compositional. ind_offset gets us beyond radhydro species.
  const auto ind_ni = species_indexer->get<int>("ni56");
  const auto ind_co = species_indexer->get<int>("co56");
  const auto ind_fe = species_indexer->get<int>("fe56");

  // This can probably be simplified.
  Kokkos::parallel_for(
      "ni :: Update",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int ix, const int k) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(ix, iN);
          const double sqrt_gm = grid.get_sqrt_gm(X);
          const double weight = grid.get_weights(iN);
          const double X_ni =
              basis_->basis_eval(mass_fractions, ix, ind_ni, iN + 1);
          const double X_co =
              basis_->basis_eval(mass_fractions, ix, ind_co, iN + 1);
          const double f_dep =
              this->template deposition_function<Model>(ix, iN);
          const double e_ni = eps_nickel2(X_ni, X_co);
          local_sum += e_ni * f_dep * sqrt_gm * weight;
        }

        dU(ix, k, 2) +=
            local_sum * grid.get_widths(ix) / basis_->get_mass_matrix(ix, k);
      });

  // TODO(astrobarker): Should this be an option?
  const auto ind_offset = ucf.extent(2);
  Kokkos::parallel_for(
      "NI :: Decay network", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        double local_sum_ni = 0.0;
        double local_sum_co = 0.0;
        double local_sum_fe = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(ix, iN);
          const double sqrt_gm = grid.get_sqrt_gm(X);
          const double weight = grid.get_weights(iN);
          const double X_ni =
              basis_->basis_eval(mass_fractions, ix, ind_ni, iN + 1);
          const double X_co =
              basis_->basis_eval(mass_fractions, ix, ind_co, iN + 1);
          const double X_fe =
              basis_->basis_eval(mass_fractions, ix, ind_fe, iN + 1);
          local_sum_ni -= LAMBDA_NI_ * X_ni * sqrt_gm * weight;
          local_sum_co +=
              (LAMBDA_NI_ * X_ni - LAMBDA_CO_ * X_co) * sqrt_gm * weight;
          local_sum_fe += LAMBDA_CO_ * X_co * sqrt_gm * weight;
        }
        const double dx_o_mkk =
            grid.get_widths(ix) / basis_->get_mass_matrix(ix, 0);

        // Decay only alters cell average mass fractions!
        dU(ix, 0, ind_offset + ind_ni) += local_sum_ni * dx_o_mkk;
        dU(ix, 0, ind_offset + ind_co) += local_sum_co * dx_o_mkk;
        dU(ix, 0, ind_offset + ind_fe) += local_sum_fe * dx_o_mkk;
      });
}
KOKKOS_FUNCTION
template <NiHeatingModel Model>
auto NiHeatingPackage::deposition_function(const int ix, const int node) const
    -> double {
  double f_dep = 0.0;
  if constexpr (Model == NiHeatingModel::FullTrapping) {
    f_dep = 1.0;
  }
  return f_dep;
}

/**
 * @brief Nickel 56 heating timestep restriction
 * @note We simply require the timestep to be smaller than the 56Ni mean
 *lifetime.
 **/
KOKKOS_FUNCTION
auto NiHeatingPackage::min_timestep(const State *const /*state*/,
                                    const GridStructure & /*grid*/,
                                    const TimeStepInfo & /*dt_info*/) const
    -> double {
  // We limit explicit timesteps to be no larger than the Ni56 half life.
  static constexpr double MAX_DT = 6.075 * constants::seconds_to_days;
  static constexpr double dt_out = MAX_DT;
  return dt_out;
}

void NiHeatingPackage::fill_derived(State * /*state*/,
                                    const GridStructure & /*grid*/,
                                    const TimeStepInfo & /*dt_info*/) const {}

[[nodiscard]] KOKKOS_FUNCTION auto NiHeatingPackage::name() const noexcept
    -> std::string_view {
  return "NickelHeating";
}

[[nodiscard]] KOKKOS_FUNCTION auto NiHeatingPackage::is_active() const noexcept
    -> bool {
  return active_;
}

KOKKOS_FUNCTION
void NiHeatingPackage::set_active(const bool active) { active_ = active; }

} // namespace ni
