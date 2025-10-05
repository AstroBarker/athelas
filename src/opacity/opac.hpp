/**
 * @file opac.hpp
 * --------------
 *
 * @brief Declares concrete opacity model classes that implement the OpacBase
 *        interface
 *
 * @details This header defines specific opacity model implementations that
 *          inherit from the OpacBase template class. It serves as the central
 *          declaration point for all opacity model classes in the codebase.
 *
 *          We provide the following opacity models:
 *          - Constant: A simple model with constant opacity value
 *          - PowerlawRho: \kappa = k rho^exp
 *
 */

#pragma once

#include "opacity/opac_base.hpp"

namespace athelas {

class Constant : public OpacBase<Constant> {
 public:
  Constant() = default;
  explicit Constant(double kP, double kR) : kP_(kP), kR_(kR) {}

  auto planck_mean(double rho, double T, double X, double Y, double Z,
                   double *lambda) const -> double;

  auto rosseland_mean(double rho, double T, double X, double Y, double Z,
                      double *lambda) const -> double;

 private:
  double kP_{};
  double kR_{};
};

class PowerlawRho : public OpacBase<PowerlawRho> {
 public:
  PowerlawRho() = default;
  PowerlawRho(double kP, double kR, double exp) : kP_(kP), kR_(kR), exp_(exp) {}

  auto planck_mean(double rho, double T, double X, double Y, double Z,
                   double *lambda) const -> double;

  auto rosseland_mean(double rho, double T, double X, double Y, double Z,
                      double *lambda) const -> double;

 private:
  double kP_{};
  double kR_{};
  double exp_{};
};

} // namespace athelas
