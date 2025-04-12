#ifndef PROBLEM_IN_HPP_
#define PROBLEM_IN_HPP_
/**
 * @file problem_in.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Class for loading input deck
 *
 * @details Loads input deck in TOML format.
 *          See: https://github.com/marzer/tomlplusplus
 */

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

  int nlim; // number of cycles
  int ncycle_out; // std output
  Real dt_hdf5; // hdf5 output
  Real dt_init_frac; // ramp up dt

  std::string eos_type;
  Real ideal_gamma;

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
  Real b_tvd;
  Real m_tvb;
  std::string limiter_type;
  bool do_limiter;

  // opac
  std::string opac_type;

  toml::table in_table;
};

#endif // PROBLEMIN_HPP_
