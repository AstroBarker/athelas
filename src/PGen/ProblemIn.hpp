#ifndef _PROBLEMIN_HPP_
#define _PROBLEMIN_HPP_

/**
 * File     :  ProblemIn.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the problem intialization
 *
 **/

#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Error.hpp"
#include "Geometry.hpp"
#include "toml.hpp"

/* hold various program options */
struct Options {
  bool do_rad  = false;
  bool do_grav = false;
  bool restart = false;

  std::string BC = "Homogenous";

  geometry::Geometry geom    = geometry::Planar;
  PolyBasis::PolyBasis basis = PolyBasis::Legendre;
};

class ProblemIn {

 public:
  ProblemIn( const std::string fn );

  std::string ProblemName;
  std::string BC;

  int nElements;
  int nNodes;
  int nGhost;

  int pOrder;
  int tOrder;
  int nStages;

  Real xL;
  Real xR;
  Real CFL;

  Real t_end;

  PolyBasis::PolyBasis Basis;
  geometry::Geometry Geometry;
  bool Restart;
  bool do_rad;

  Real alpha;
  Real SL_Threshold;
  bool TCI_Option;
  Real TCI_Threshold;
  bool Characteristic;

  toml::table in_table;
};

#endif // _PROBLEMIN_HPP_
