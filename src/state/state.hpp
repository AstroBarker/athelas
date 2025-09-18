#pragma once

#include <memory>

#include "composition/compdata.hpp"
#include "utils/abstractions.hpp"

/**
 * @class State
 * @brief Primary State datastructure.
 */
class State {
 public:
  State(int nvar, int nPF, int nAF, int nX_, int nNodes_, int pOrder,
        int nstages, bool composition_enabled, bool ionization_enabled);

  [[nodiscard]] auto n_cf() const noexcept -> int;
  [[nodiscard]] auto n_pf() const noexcept -> int;
  [[nodiscard]] auto n_af() const noexcept -> int;
  [[nodiscard]] auto p_order() const noexcept -> int;

  [[nodiscard]] auto u_cf() const noexcept -> View3D<double>;
  [[nodiscard]] auto u_cf_stages() const noexcept -> View4D<double>;
  [[nodiscard]] auto u_pf() const noexcept -> View3D<double>;
  [[nodiscard]] auto u_af() const noexcept -> View3D<double>;

  [[nodiscard]] auto composition_enabled() const noexcept -> bool;
  [[nodiscard]] auto ionization_enabled() const noexcept -> bool;

  [[nodiscard]] auto comps() const -> CompositionData *;
  [[nodiscard]] auto ionization_state() const -> IonizationState *;

  void setup_composition(std::shared_ptr<CompositionData> comps);
  void setup_ionization(std::shared_ptr<IonizationState> ion);

 private:
  int nvar_;
  int nPF_;
  int nAF_;
  int pOrder_;

  View3D<double> uCF_; // Conserved fluid
  View4D<double> uCF_s_; // Conserved fluid (stage storage)
  View3D<double> uPF_; // primitive fluid
  View3D<double> uAF_; // auxiliary fluid

  std::shared_ptr<CompositionData> comps_;
  std::shared_ptr<IonizationState> ionization_state_;

  bool composition_enabled_;
  bool ionization_enabled_;
};
