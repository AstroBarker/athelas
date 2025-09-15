#pragma once
/**
 * @file opac_variant.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Provides variant-based dispatch for opacity model operations
 *
 * @details This header implements a type-safe way to handle different opacity
 *          models at runtime using std::variant. It provides visitor functions
 *          that dispatch to the appropriate model's implementation.
 */

#include <variant>

#include "opac.hpp"
#include "problem_in.hpp"

using Opacity = std::variant<Constant, PowerlawRho>;

KOKKOS_INLINE_FUNCTION auto planck_mean(const Opacity *opac, const double rho,
                                        const double T, const double X,
                                        const double Y, const double Z,
                                        double *lambda) -> double {
  return std::visit(
      [&rho, &T, &X, &Y, &Z, &lambda](auto &opac) {
        return opac.planck_mean(rho, T, X, Y, Z, lambda);
      },
      *opac);
}

KOKKOS_INLINE_FUNCTION auto
rosseland_mean(const Opacity *opac, const double rho, const double T,
               const double X, const double Y, const double Z, double *lambda)
    -> double {
  return std::visit(
      [&rho, &T, &X, &Y, &Z, &lambda](auto &opac) {
        return opac.rosseland_mean(rho, T, X, Y, Z, lambda);
      },
      *opac);
}

// put init function here..

KOKKOS_INLINE_FUNCTION auto initialize_opacity(const ProblemIn *pin)
    -> Opacity {
  Opacity opac;
  const auto type = pin->param()->get<std::string>("opac.type", "constant");
  if (type == "constant") {
    opac = Constant(pin->param()->get<double>("opac.kP", 1.0),
                    pin->param()->get<double>("opac.kR", 1.0));
  } else { // powerlaw rho
    opac = PowerlawRho(pin->param()->get<double>("opac.kP", 1.0),
                       pin->param()->get<double>("opac.kR", 1.0),
                       pin->param()->get<double>("opac.exp", 1.0));
  }
  return opac;
}
