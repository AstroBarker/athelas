#ifndef TABLEAU_HPP_
#define TABLEAU_HPP_
/**
 * @file tableau.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Class for holding implicit and explicit RK tableaus.
 *
 * @details TODO: describe tableaus.
 */

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "error.hpp"

enum TableauType { Implicit, Explicit };

/**
 * @brief Butcher tableau class.
 **/
class ButcherTableau {
 public:
  ButcherTableau( int nStages_, int tOrder_, TableauType type );

  int nStages;
  int tOrder;

  View2D<Real> a_ij{ };
  View1D<Real> b_i{ };

 private:
  TableauType type_;
  int c_effective_{ };

  void initialize_tableau( );
};

/**
 * @brief Shu Osher tableau class.
 **/
class ShuOsherTableau {
 public:
  ShuOsherTableau( int nStages, int tOrder, TableauType type );

  int nStages;
  int tOrder;

  View2D<Real> a_ij{ };
  View2D<Real> b_ij{ };

 private:
  TableauType type_;
  int c_effective_{ };

  void initialize_tableau( );
};

#endif // TABLEAU_HPP_
