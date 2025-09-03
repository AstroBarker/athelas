#pragma once

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * Initialize one_zone_ionization test
 **/
void one_zone_ionization_init(State* state, GridStructure* grid, ProblemIn* pin,
                              const EOS* eos,
                              ModalBasis* /*fluid_basis = nullptr*/) {
  const bool ionization_active =
      pin->param()->get<bool>("physics.ionization_enabled");
  const int saha_ncomps =
      pin->param()->get<int>("ionization.ncomps"); // for ionization
  const auto ncomps =
      pin->param()->get<int>("problem.params.ncomps", 1); // mass fractions
  if (!ionization_active) {
    THROW_ATHELAS_ERROR("One zone ionization requires ionization enabled!");
  }
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("One zone ionization requires ideal gas eos!");
  }
  // Don't try to track ionization for more species than we use.
  // We will track ionization for the first saha_ncomps species
  if (saha_ncomps > ncomps) {
    THROW_ATHELAS_ERROR("One zone ionization requires [ionization.ncomps] >= "
                        "[problem.params.ncomps]!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();

  const int ilo = 1;
  const int ihi = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();

  const int q_Tau = 0;
  const int q_V = 1;
  const int q_E = 2;

  const int iPF_D = 0;

  const auto temperature =
      pin->param()->get<double>("problem.params.temperature", 5800); // K
  const auto rho =
      pin->param()->get<double>("problem.params.rho", 1000.0); // g/cc
  const double vel = 0.0;
  const double tau = 1.0 / rho;

  const auto fn_ionization =
      pin->param()->get<std::string>("ionization.fn_ionization");
  const auto fn_deg =
      pin->param()->get<std::string>("ionization.fn_degeneracy");

  if (temperature <= 0.0 || rho <= 0.0) {
    THROW_ATHELAS_ERROR("Temperature and denisty must be positive definite!");
  }

  const double mu = 1.0 + constants::m_e / constants::m_p;
  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;
  const double sie = constants::k_B * temperature / (gm1 * mu * constants::m_p);

  std::shared_ptr<CompositionData> comps = std::make_shared<CompositionData>(
      grid->get_n_elements() + 2, nNodes, ncomps);
  std::shared_ptr<IonizationState> ionization_state =
      std::make_shared<IonizationState>(grid->get_n_elements() + 2, nNodes,
                                        saha_ncomps, saha_ncomps + 1,
                                        fn_ionization, fn_deg);
  auto mass_fractions = comps->mass_fractions();
  auto charges = comps->charge();
  auto ionization_states = ionization_state->ionization_fractions();
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 2), KOKKOS_LAMBDA(int ix) {
        const int k = 0;

        uCF(ix, k, q_Tau) = tau;
        uCF(ix, k, q_V) = vel;
        uCF(ix, k, q_E) = sie;

        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          uPF(ix, iNodeX, iPF_D) = rho;
        }

        // set up comps
        // For this problem we set up a contiguous list of species
        // form Z = 1 to ncomps. Mass fractions are uniform.
        for (int node = 0; node < nNodes; ++node) {
          for (int elem = 0; elem < ncomps; ++elem) {
            charges(elem) = elem + 1;
            mass_fractions(ix, node, elem) = 1.0 / ncomps;
          }

          // overkill
          if (ionization_active) {
            for (int elem = 0; elem < saha_ncomps; ++elem) {
              const int Z = charges(elem);

              for (int z = 0; z < Z + 1; ++z) {
                ionization_states(ix, node, elem, z) = 0.0; // unnecessary
              }
            }
          }
        }
      });

  state->setup_composition(comps);
  state->setup_ionization(ionization_state);

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(ilo - 1 - ix, iN, 0) = uPF(ilo + ix, nNodes - iN - 1, 0);
          uPF(ilo + 1 + ix, iN, 0) = uPF(ilo - ix, nNodes - iN - 1, 0);
        }
      });
}
