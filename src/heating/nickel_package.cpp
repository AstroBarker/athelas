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
  static const int &ihi = grid.get_ihi();
  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  auto *comps = state->comps();
  const auto *const species_indexer = comps->species_indexer();
  // index gymnastics. dU holds updates for all quantities including
  // compositional. ind_offset gets us beyond radhydro species.
  const auto ind_ni = species_indexer->get<int>("ni56");
  const auto ind_co = species_indexer->get<int>("co56");
  const auto ind_fe = species_indexer->get<int>("fe56");
  const auto ind_offset = ucf.extent(2);

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "NiHeating :: Zero dU", ihi + 1, KOKKOS_LAMBDA(const int ix) {
        dU(ix, 0, ind_offset + ind_ni) = 0.0;
        dU(ix, 0, ind_offset + ind_co) = 0.0;
        dU(ix, 0, ind_offset + ind_fe) = 0.0;
      });

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

  const auto mass = grid.mass();
  const auto mass_fractions_stages = comps->mass_fractions_stages();
  const auto mass_fractions =
      Kokkos::subview(mass_fractions_stages, dt_info.stage, Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL);
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
          const double weight = grid.get_weights(iN);
          const double X_ni =
              basis_->basis_eval(mass_fractions, ix, ind_ni, iN + 1);
          const double X_co =
              basis_->basis_eval(mass_fractions, ix, ind_co, iN + 1);
          const double f_dep =
              this->template deposition_function<Model>(ix, iN);
          const double e_ni = eps_nickel2(X_ni, X_co);
          // const double e_ni = eps_nickel1(time);
          local_sum +=
              e_ni * f_dep * 1.0 * weight * basis_->get_phi(ix, iN + 1, k);
        }

        const double dx_o_mkk = mass(ix) / basis_->get_mass_matrix(ix, k);
        dU(ix, k, 2) += local_sum * dx_o_mkk;
        //                        std::println("dU e {}", dU(ix, k, 2));
      });

  // TODO(astrobarker): Should this be an option?
  // NOTE: Nickel decay chain only affects cell averages.
  // Realistically I don't need to integrate X_Fe, but oh well.
  const auto ind_offset = ucf.extent(2);
  Kokkos::parallel_for(
      "NI :: Decay network", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        const double x_ni = mass_fractions(ix, 0, ind_ni);
        const double x_co = mass_fractions(ix, 0, ind_co);
        const double rhs_ni = -LAMBDA_NI_ * x_ni;
        const double rhs_co = LAMBDA_NI_ * x_ni - LAMBDA_CO_ * x_co;
        const double rhs_fe = LAMBDA_CO_ * x_co;

        // Decay only alters cell average mass fractions!
        dU(ix, 0, ind_offset + ind_ni) += rhs_ni;
        dU(ix, 0, ind_offset + ind_co) += rhs_co;
        dU(ix, 0, ind_offset + ind_fe) += rhs_fe;
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
  // We limit explicit timesteps to be no larger than the 1/10 Ni56 mean
  // lifetime
  static constexpr double MAX_DT = TAU_NI_ / 10.0;
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
