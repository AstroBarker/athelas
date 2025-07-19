#pragma once
/**
 * @file eos.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Declares equation of state classes that implement the EosBase
 * interface
 *
 * @details Defines specific equation of state implementations that inherit
 *          from the EosBase template class. It serves as the central
 * declaration point for all EOS classes in the codebase, with their
 * implementations provided in separate .cpp files.
 *
 *          We support the following equations of state:
 *          - IdealGas (default): ideal gas EOS
 *          - Marshak: Weird ideal gas for Marshak test.
 *          - Stellar: currently an unused placceholder
 *
 *          Note: implementations are supplied in eos-specific cpp files.
 */

#include "eos_base.hpp"
#include "error.hpp"

class IdealGas : public EosBase<IdealGas> {
 public:
  IdealGas() = default;
  explicit IdealGas(double gm) : gamma_(gm) {
    if (gamma_ <= 0.0) {
      THROW_ATHELAS_ERROR(" ! IdealGas :: Adiabatic gamma <= 0.0!");
    }
  }

  auto pressure_from_conserved(double tau, double V, double EmT,
                               double* lambda) const -> double;
  auto sound_speed_from_conserved(double tau, double V, double EmT,
                                  double* lambda) const -> double;
  auto temperature_from_conserved(double tau, double V, double E,
                                  double* lambda) const -> double;
  [[nodiscard]] auto get_gamma() const noexcept -> double;

 private:
  double gamma_{};
};

// TODO(astrobarker): thread su olson alpha in
class Marshak : public EosBase<Marshak> {
 public:
  Marshak() = default;
  explicit Marshak(double gm) : gamma_(gm) {
    if (gamma_ <= 0.0) {
      THROW_ATHELAS_ERROR(" ! Marshak :: Adiabatic gamma <= 0.0!");
    }
  }

  auto pressure_from_conserved(double tau, double V, double EmT,
                               double* lambda) const -> double;
  auto sound_speed_from_conserved(double tau, double V, double EmT,
                                  double* lambda) const -> double;
  auto temperature_from_conserved(double tau, double V, double E,
                                  double* lambda) const -> double;
  [[nodiscard]] auto get_gamma() const noexcept -> double;

 private:
  double gamma_{};
};

/* placeholder */
class Stellar : public EosBase<Stellar> {
 public:
  Stellar() = default;

  auto pressure_from_conserved(double tau, double V, double EmT,
                               double* lambda) const -> double;
  auto sound_speed_from_conserved(double tau, double V, double EmT,
                                  double* lambda) const -> double;
  auto temperature_from_conserved(double tau, double V, double E,
                                  double* lambda) const -> double;

 private:
  double gamma_{};
};
