#pragma once
/**
 * @file opac_base.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Base class for opacity models.
 *
 * @details Defines the OpacBase template class.
 *
 *          The class provides two interface methods:
 *          - planck_mean
 *          - rosseland_mean
 *
 *          The interface methods take density, temperature, and composition
 *          parameters to compute the appropriate mean opacity values.
 */

template <class OPAC>
class OpacBase {
 public:
  auto planck_mean(const double rho, const double T, const double X,
                   const double Y, const double Z, double* lambda) const
      -> double {
    return static_cast<OPAC const*>(this)->planck_mean(rho, T, X, Y, Z, lambda);
  }

  auto rosseland_mean(const double rho, const double T, const double X,
                      const double Y, const double Z, double* lambda) const
      -> double {
    return static_cast<OPAC const*>(this)->rosseland_mean(rho, T, X, Y, Z,
                                                          lambda);
  }
};
