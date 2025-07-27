#pragma once
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

#include "abstractions.hpp"
#include "error.hpp"
#include "geometry.hpp"
#include "timestepper/tableau.hpp"
#include "toml.hpp"

/* hold various program options */
struct Options {
  bool do_rad  = false;
  bool do_grav = false;
  bool restart = false;

  geometry::Geometry geom      = geometry::Planar;
  poly_basis::poly_basis basis = poly_basis::legendre;

  int max_order = 1;
};

// TODO(astrobarker): Long term solution for this thing.
// "Params" style wrapper over in_table with GetOrAdd?
class ProblemIn {

 public:
  explicit ProblemIn(const std::string& fn);

  std::string problem_name;
  std::string fluid_bc_i;
  std::string fluid_bc_o;
  std::string rad_bc_i;
  std::string rad_bc_o;
  std::array<double, 3> fluid_i_dirichlet_values;
  std::array<double, 3> fluid_o_dirichlet_values;
  std::array<double, 2> rad_i_dirichlet_values;
  std::array<double, 2> rad_o_dirichlet_values;

  int nlim{}; // number of cycles
  int ncycle_out{}; // std output
  double dt_hdf5{}; // hdf5 output
  double dt_init_frac{}; // ramp up dt

  std::string eos_type;

  int nElements;
  int nNodes;
  int nGhost;

  int pOrder;
  int tOrder;
  int nStages;
  std::string integrator;
  MethodID method_id;

  double xL;
  double xR;
  double CFL;

  double t_end;

  poly_basis::poly_basis basis;
  geometry::Geometry Geometry;
  bool Restart;
  bool do_rad;

  bool TCI_Option;
  double TCI_Threshold;
  bool Characteristic;
  double gamma_l;
  double gamma_i;
  double gamma_r;
  double weno_r;
  double b_tvd{};
  double m_tvb{};
  std::string limiter_type;
  bool do_limiter{};

  // opac
  std::string opac_type;

  bool history_enabled;
  std::string hist_fn;
  double hist_dt;

  // gravity
  bool do_gravity;
  GravityModel grav_model;
  double gval;

  toml::table in_table;
};

// TODO(astrobarker) move into class
auto check_bc(std::string& bc) -> bool;
template <typename T, typename G>
void read_toml_array(T toml_array, G& out_array) {
  long unsigned int index = 0;
  for (const auto& element : *toml_array) {
    if (index < out_array.size()) {
      if (auto elem = element.as_floating_point()) {
        out_array[index] = static_cast<double>(*elem);
      } else {
        std::cerr << "Type mismatch at index " << index << "\n";
        THROW_ATHELAS_ERROR(" ! Error reading dirichlet boundary conditions.");
      }
      index++;
    } else {
      std::cerr << "TOML array is larger than C++ array." << "\n";
      THROW_ATHELAS_ERROR(" ! Error reading dirichlet boundary conditions.");
    }
  }
}
