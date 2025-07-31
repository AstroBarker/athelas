#pragma once
/**
 * @file eos_variant.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Provides variant-based dispatch for equations of state
 *
 * @details This header implements a type-safe way to handle different EOS
 *          models at runtime using std::variant. It provides visitor functions
 *          that dispatch to the appropriate model's implementation.
 */

#include <variant>

#include "eos.hpp"
#include "error.hpp"
#include "problem_in.hpp"

using EOS = std::variant<IdealGas, Marshak>;

KOKKOS_INLINE_FUNCTION auto
pressure_from_conserved(const EOS* eos, const double tau, const double V,
                        const double E, double* lambda) -> double {
  return std::visit(
      [&tau, &V, &E, &lambda](auto& eos) {
        return eos.pressure_from_conserved(tau, V, E, lambda);
      },
      *eos);
}

KOKKOS_INLINE_FUNCTION auto
sound_speed_from_conserved(const EOS* eos, const double tau, const double V,
                           const double E, double* lambda) -> double {
  return std::visit(
      [&tau, &V, &E, &lambda](auto& eos) {
        return eos.sound_speed_from_conserved(tau, V, E, lambda);
      },
      *eos);
}

KOKKOS_INLINE_FUNCTION auto
temperature_from_conserved(const EOS* eos, const double tau, const double V,
                           const double E, double* lambda) -> double {
  return std::visit(
      [&tau, &V, &E, &lambda](auto& eos) {
        return eos.temperature_from_conserved(tau, V, E, lambda);
      },
      *eos);
}

KOKKOS_INLINE_FUNCTION auto get_gamma(const EOS* eos) -> double {
  return std::visit([](auto& eos) { return eos.get_gamma(); }, *eos);
}

KOKKOS_INLINE_FUNCTION auto initialize_eos(const ProblemIn* pin) -> EOS {
  EOS eos;
  const auto type = pin->param()->get<std::string>("eos.type");
  if (type == "ideal") {
    eos = IdealGas(pin->param()->get<double>("eos.ideal_gamma"));
  } else if (type == "marshak") {
    eos = Marshak(pin->param()->get<double>("eos.ideal_gamma"));
  } else {
    THROW_ATHELAS_ERROR("Please choose a valid eos!");
  }
  return eos;
}
