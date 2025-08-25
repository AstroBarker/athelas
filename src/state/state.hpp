#pragma once

#include "utils/abstractions.hpp"
#include "utils/error.hpp"

/**
 * @class State
 * @brief Primary State datastructure.
 */
class State {
 public:
  State(int nvar, int nPF, int nAF, int nX_, int nNodes_, int pOrder,
        bool composition_enabled, int ncomps = 0);

  [[nodiscard]] auto n_cf() const noexcept -> int;
  [[nodiscard]] auto n_pf() const noexcept -> int;
  [[nodiscard]] auto n_af() const noexcept -> int;
  [[nodiscard]] auto p_order() const noexcept -> int;

  [[nodiscard]] auto u_cf() const noexcept -> View3D<double>;
  [[nodiscard]] auto u_pf() const noexcept -> View3D<double>;
  [[nodiscard]] auto u_af() const noexcept -> View3D<double>;

  [[nodiscard]] auto has_composition() const noexcept -> bool {
    return composition_enabled_;
  }

  [[nodiscard]] auto u_comp() const -> View3D<double> {
    if (!composition_enabled_) {
      THROW_ATHELAS_ERROR("Composition not enabled!");
    }
    return uComp_;
  }

 private:
  int nvar_;
  int nPF_;
  int nAF_;
  int pOrder_;

  View3D<double> uCF_; // Conserved fluid
  View3D<double> uPF_; // primitive fluid
  View3D<double> uAF_; // auxiliary fluid
  View3D<double> uComp_; // Composition. Default constructed = no allocation

  bool composition_enabled_;
};
