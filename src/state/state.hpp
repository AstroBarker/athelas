#pragma once

#include <memory>

#include "composition/compdata.hpp"
#include "interface/params.hpp"
#include "kokkos_types.hpp"
#include "pgen/problem_in.hpp"

namespace athelas {

/**
 * @class State
 * @brief Primary State datastructure.
 */
class State {
 public:
  State(const ProblemIn *pin, int nstages);

  [[nodiscard]] auto n_cf() const noexcept -> int;
  [[nodiscard]] auto n_pf() const noexcept -> int;
  [[nodiscard]] auto n_af() const noexcept -> int;
  [[nodiscard]] auto p_order() const noexcept -> int;

  [[nodiscard]] auto u_cf() const noexcept -> AthelasArray3D<double>;
  [[nodiscard]] auto u_cf_stages() const noexcept -> AthelasArray4D<double>;
  [[nodiscard]] auto u_pf() const noexcept -> AthelasArray3D<double>;
  [[nodiscard]] auto u_af() const noexcept -> AthelasArray3D<double>;

  [[nodiscard]] auto composition_enabled() const noexcept -> bool;
  [[nodiscard]] auto ionization_enabled() const noexcept -> bool;
  [[nodiscard]] auto composition_evolved() const noexcept -> bool;
  [[nodiscard]] auto nickel_evolved() const noexcept -> bool;

  [[nodiscard]] auto comps() const -> atom::CompositionData *;
  [[nodiscard]] auto ionization_state() const -> atom::IonizationState *;

  void setup_composition(std::shared_ptr<atom::CompositionData> comps);
  void setup_ionization(std::shared_ptr<atom::IonizationState> ion);

  auto params() noexcept -> Params *;

 private:
  std::unique_ptr<Params> params_;

  AthelasArray3D<double> uCF_; // Conserved fluid
  AthelasArray4D<double> uCF_s_; // Conserved fluid (stage storage)
  AthelasArray3D<double> uPF_; // primitive fluid
  AthelasArray3D<double> uAF_; // auxiliary fluid

  std::shared_ptr<atom::CompositionData> comps_;
  std::shared_ptr<atom::IonizationState> ionization_state_;
};

} // namespace athelas
