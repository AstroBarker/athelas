#ifndef TABLEAU_HPP_
#define TABLEAU_HPP_

/**
 * File     :  tableau.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Classes for holding Runge-Kutta tableaus
 *
 **/

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "error.hpp"

enum TableauType { Implicit, Explicit };

/**
 * Butcher tableau class.
 **/
class ButcherTableau {
 public:
  ButcherTableau( const int nStages_, const int tOrder_,
                  const TableauType type );

  int nStages;
  int tOrder;

  View2D<Real> a_ij;
  View1D<Real> b_i;

 private:
  TableauType type_;
  int c_effective_;

  void initialize_tableau( );
};

/**
 * Shu Osher tableau class.
 **/
class ShuOsherTableau {
 public:
  ShuOsherTableau( const int nStages, const int tOrder,
                   const TableauType type );

  int nStages;
  int tOrder;

  View2D<Real> a_ij;
  View2D<Real> b_ij;

 private:
  TableauType type_;
  int c_effective_;

  void initialize_tableau( );
};

#endif // TABLEAU_HPP_
