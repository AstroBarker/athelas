#pragma once

#include <memory>

#include "composition/composition.hpp"
#include "utils/abstractions.hpp"

/**
 * @class State
 * @brief Primary State datastructure.
 */
class State {
 public:
  State(int nvar, int nPF, int nAF, int nX_, int nNodes_, int pOrder,
        bool composition_enabled);

  [[nodiscard]] auto n_cf() const noexcept -> int;
  [[nodiscard]] auto n_pf() const noexcept -> int;
  [[nodiscard]] auto n_af() const noexcept -> int;
  [[nodiscard]] auto p_order() const noexcept -> int;

  [[nodiscard]] auto u_cf() const noexcept -> View3D<double>;
  [[nodiscard]] auto u_pf() const noexcept -> View3D<double>;
  [[nodiscard]] auto u_af() const noexcept -> View3D<double>;

  [[nodiscard]] auto composition_enabled() const noexcept -> bool {
    return composition_enabled_;
  }

  [[nodiscard]] auto comps() const -> CompositionData*;

  void setup_composition(std::shared_ptr<CompositionData> comps);

 private:
  int nvar_;
  int nPF_;
  int nAF_;
  int pOrder_;

  View3D<double> uCF_; // Conserved fluid
  View3D<double> uPF_; // primitive fluid
  View3D<double> uAF_; // auxiliary fluid

  std::shared_ptr<CompositionData> comps_;

  bool composition_enabled_;
};
