#pragma once

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "composition/saha.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * Initialize ni_decay test
 **/
void ni_decay_init(State *state, GridStructure *grid, ProblemIn *pin,
                   const EOS *eos, ModalBasis *fluid_basis = nullptr) {
  const bool composition_active =
      pin->param()->get<bool>("physics.composition_enabled");
  const bool ni_decay_active =
      pin->param()->get<bool>("physics.heating.nickel.enabled");
  const auto ncomps = 3; // Ni, Co, Fe
  if (!composition_active) {
    THROW_ATHELAS_ERROR("Ni decay requires composition enabled!");
  }
  if (!ni_decay_active) {
    THROW_ATHELAS_ERROR("Ni decay requires nickel heating enabled!");
  }
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Ni decay requires ideal gas eos!");
  }

  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();
  auto uAF = state->u_af();

  static const int ilo = 1;
  static const int ihi = grid->get_ihi();
  static const int nNodes = grid->get_n_nodes();
  static const int order = nNodes;

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

  if (temperature <= 0.0 || rho <= 0.0) {
    THROW_ATHELAS_ERROR("Temperature and denisty must be positive definite!");
  }

  const double mu = 1.0 + constants::m_e / constants::m_p;
  const double gamma = get_gamma(eos);
  const double gm1 = gamma - 1.0;
  const double sie = constants::k_B * temperature / (gm1 * mu * constants::m_p);

  std::shared_ptr<CompositionData> comps = std::make_shared<CompositionData>(
      grid->get_n_elements() + 2, order, ncomps);
  auto mass_fractions = comps->mass_fractions();
  auto charges = comps->charge();
  auto neutrons = comps->neutron_number();
  auto *species_indexer = comps->species_indexer();
  species_indexer->add("ni56", 0);
  species_indexer->add("co56", 1);
  species_indexer->add("fe56", 2);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ihi + 2), KOKKOS_LAMBDA(int ix) {
        const int k = 0;

        uCF(ix, k, q_Tau) = tau;
        uCF(ix, k, q_V) = vel;
        uCF(ix, k, q_E) = sie;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(ix, iNodeX, iPF_D) = rho;
          uAF(ix, iNodeX, 1) = temperature;
        }

        // set up comps
        // For this problem we set up a contiguous list of species
        // form Z = 1 to ncomps. Mass fractions are uniform with no slopes.
        mass_fractions(ix, k, 0) = 1.0; // Pure Ni

        // Ni
        charges(0) = 28;
        neutrons(0) = 28;
        // Co
        charges(1) = 27;
        neutrons(1) = 29;
        // Co
        charges(2) = 26;
        neutrons(2) = 30;
      });

  state->setup_composition(comps);

  // Fill density in guard cells
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ilo), KOKKOS_LAMBDA(int ix) {
        for (int iN = 0; iN < nNodes; iN++) {
          uPF(ilo - 1 - ix, iN, 0) = uPF(ilo + ix, nNodes - iN - 1, 0);
          uPF(ilo + 1 + ix, iN, 0) = uPF(ilo - ix, nNodes - iN - 1, 0);
        }
      });
}
