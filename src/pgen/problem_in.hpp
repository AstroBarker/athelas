#ifndef _PROBLEM_IN_HPP_
#define _PROBLEM_IN_HPP_

/**
 * File     :  problem_in.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the problem intialization
 *
 **/

#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "error.hpp"
#include "geometry.hpp"
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

  std::string problem_name;
  std::string BC;

  int nlim;
  int ncycle_out;
  Real dt_hdf5;

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

  bool TCI_Option;
  Real TCI_Threshold;
  bool Characteristic;
  Real gamma_l;
  Real gamma_i;
  Real gamma_r;
  Real weno_r;

  toml::table in_table;
};

#endif // _PROBLEMIN_HPP_
