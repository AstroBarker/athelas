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
#include <vector>

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "error.hpp"
#include "geometry.hpp"
#include "toml.hpp"

/* hold various program options */
struct Options {
  bool do_rad  = false;
  bool do_grav = false;
  bool restart = false;

  geometry::Geometry geom      = geometry::Planar;
  poly_basis::poly_basis basis = poly_basis::legendre;
};

class ProblemIn {

 public:
  explicit ProblemIn( const std::string& fn );

  std::string problem_name;
  std::string fluid_bc_i;
  std::string fluid_bc_o;
  std::string rad_bc_i;
  std::string rad_bc_o;
  std::array<Real, 3> fluid_i_dirichlet_values;
  std::array<Real, 3> fluid_o_dirichlet_values;
  std::array<Real, 2> rad_i_dirichlet_values;
  std::array<Real, 2> rad_o_dirichlet_values;

  int nlim{ }; // number of cycles
  int ncycle_out{ }; // std output
  Real dt_hdf5{ }; // hdf5 output
  Real dt_init_frac{ }; // ramp up dt

  std::string eos_type;
  Real gamma_eos{ };

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

  poly_basis::poly_basis basis;
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
  Real b_tvd{ };
  Real m_tvb{ };
  std::string limiter_type;
  bool do_limiter{ };

  // opac
  std::string opac_type;

  toml::table in_table;
};

// TODO(astrobarker) move into class
bool check_bc( std::string& bc );
template <typename T, typename G>
void read_toml_array( T toml_array, G& out_array ) {
  long unsigned int index = 0;
  for ( const auto& element : *toml_array ) {
    if ( index < out_array.size( ) ) {
      if ( auto elem = element.as_floating_point( ) ) {
        out_array[index] = static_cast<Real>( *elem );
      } else {
        std::cerr << "Type mismatch at index " << index << std::endl;
        THROW_ATHELAS_ERROR(
            " ! Error reading dirichlet boundary conditions." );
      }
      index++;
    } else {
      std::cerr << "TOML array is larger than C++ array." << std::endl;
      THROW_ATHELAS_ERROR( " ! Error reading dirichlet boundary conditions." );
    }
  }
}
