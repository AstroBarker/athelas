/**
 * @file io.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief HDF5 and std out IO routines
 *
 * @details Collection of functions for IO using H5Cpp for HDF5 operations
 *
 * TODO(astrobarker): make device friendly
 */

#include <array>
#include <cstddef>
#include <iomanip>
#include <map>
#include <print>
#include <sstream>
#include <string>
#include <vector>

#include "H5Cpp.h"

#include "basis/polynomial_basis.hpp"
#include "build_info.hpp"
#include "geometry/grid.hpp"
#include "io/io.hpp"
#include "limiters/slope_limiter.hpp"
#include "timestepper/tableau.hpp"

/**
 * Write to standard output some initialization info
 * for the current simulation.
 **/
void print_simulation_parameters(GridStructure grid, ProblemIn* pin) {
  const int nX = grid.get_n_elements();
  const int nNodes = grid.get_n_nodes();
  // NOTE: If I properly support more bases again, adjust here.
  const std::string basis_name = "Legendre";
  const bool rad_enabled = pin->param()->get<bool>("physics.rad_active");
  const bool gravity_enabled =
      pin->param()->get<bool>("physics.gravity_active");

  std::println("# --- General --- ");
  std::println("# Problem Name    : {}",
               pin->param()->get<std::string>("problem.problem"));
  std::println("# CFL             : {}",
               pin->param()->get<double>("problem.cfl"));
  std::println("");

  std::println("# --- Grid Parameters --- ");
  std::println("# Mesh Elements  : {}", nX);
  std::println("# Number Nodes   : {}", nNodes);
  std::println("# Lower Boundary : {}", grid.get_x_l());
  std::println("# Upper Boundary : {}", grid.get_x_r());
  std::println("");

  std::println("# --- Physics Parameters --- ");
  std::println("# Radiation      : {}", rad_enabled);
  std::println("# Gravity        : {}", gravity_enabled);
  std::println("# EOS            : {}",
               pin->param()->get<std::string>("eos.type"));
  std::println("");

  std::println("# --- Discretization Parameters --- ");
  std::println("# Basis          : {}", basis_name);
  std::println("# Integrator     : {}",
               pin->param()->get<std::string>("time.integrator_string"));
  std::println("");

  std::println("# --- Fluid Parameters --- ");
  std::println("# Spatial Order  : {}", pin->param()->get<int>("fluid.porder"));
  std::println("# Inner BC       : {}",
               pin->param()->get<std::string>("fluid.bc.i"));
  std::println("# Outer BC       : {}",
               pin->param()->get<std::string>("fluid.bc.o"));
  std::println("");

  std::println("# --- Fluid Limiter --- ");
  if (pin->param()->get<int>("fluid.porder") == 1) {
    std::println("# Spatial Order 1: Slope limiter not applied.");
  }
  if (!pin->param()->get<bool>("fluid.limiter.enabled")) {
    std::println("# Limiter Disabled");
  } else {
    const auto limiter_type =
        pin->param()->get<std::string>("fluid.limiter.type");
    std::println("# Limiter        : {}", limiter_type);
  }
  std::println("");

  if (rad_enabled) {
    std::println("# --- Radiation Parameters --- ");
    std::println("# Spatial Order  : {}",
                 pin->param()->get<int>("radiation.porder"));
    std::println("# Inner BC       : {}",
                 pin->param()->get<std::string>("radiation.bc.i"));
    std::println("# Outer BC       : {}",
                 pin->param()->get<std::string>("radiation.bc.o"));
    std::println("");

    std::println("# --- Radiation Limiter Parameters --- ");
    if (pin->param()->get<int>("radiation.porder") == 1) {
      std::println("# Spatial Order 1: Slope limiter not applied.");
    }
    if (!pin->param()->get<bool>("radiation.limiter.enabled")) {
      std::println("# Limiter Disabled");
    } else {
      const auto limiter_type =
          pin->param()->get<std::string>("radiation.limiter.type");
      std::println("# Limiter        : {}", limiter_type);
    }
    std::println("");
  }

  if (gravity_enabled) {
    std::println("# --- Gravity Parameters --- ");
    std::println("# Modal           : {}",
                 pin->param()->get<std::string>("gravity.modelstring"));
    std::println("");
  }
}

/**
 * Write simulation output to disk
 **/

// Structure to hold variable metadata
struct VariableInfo {
  std::string path;
  std::string description;
  bool is_modal; // true if variable has modal structure (nX * order), false if
                 // cell-centered (nX)
};

// Helper class for HDF5 output management
class HDF5Writer {
 private:
  H5::H5File file_;
  std::map<std::string, H5::Group> groups_;

 public:
  HDF5Writer(const std::string& filename) : file_(filename, H5F_ACC_TRUNC) {}

  // Create group hierarchy
  void createGroup(const std::string& path) {
    if (groups_.find(path) != groups_.end()) return;

    // Create parent groups recursively
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos && pos > 0) {
      std::string parent = path.substr(0, pos);
      createGroup(parent);
    }

    groups_[path] = file_.createGroup(path);
  }

  // Write scalar metadata
  template <typename T>
  void write_scalar(const std::string& path, const T& value,
                    const H5::DataType& h5type) {
    std::array<hsize_t, 1> dim = {1};
    H5::DataSpace space(1, dim.data());
    H5::DataSet dataset = file_.createDataSet(path, h5type, space);
    dataset.write(&value, h5type);
  }

  // Write string metadata
  void write_string(const std::string& path, const std::string& value) {
    std::array<hsize_t, 1> dim = {1};
    H5::DataSpace space(1, dim.data());
    H5::StrType stringtype(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSet dataset = file_.createDataSet(path, stringtype, space);
    dataset.write(value, stringtype);
  }

  // Write vector data
  void write_vector(const std::string& path, const std::vector<double>& data,
                    const std::string& description = "") {
    std::array<hsize_t, 1> dim = {data.size()};
    H5::DataSpace space(1, dim.data());
    H5::DataSet dataset =
        file_.createDataSet(path, H5::PredType::NATIVE_DOUBLE, space);
    dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);

    if (!description.empty()) {
      H5::StrType str_type(H5::PredType::C_S1, description.length());
      H5::Attribute attr = dataset.createAttribute("description", str_type,
                                                   H5::DataSpace(H5S_SCALAR));
      attr.write(str_type, description);
    }
  }
};

// Generate filename with proper padding
auto generate_filename(const std::string& problem_name, int i_write,
                       int max_digits = 4) -> std::string {
  std::ostringstream oss;
  oss << problem_name << "_";

  if (i_write != -1) {
    oss << std::setfill('0') << std::setw(max_digits) << i_write;
  } else {
    oss << "final";
  }

  oss << ".h5";
  return oss.str();
}

// @brief write to hdf5
void write_state(State* state, GridStructure& grid, SlopeLimiter* SL,
                 ProblemIn* pin, double time, int order, int i_write,
                 bool do_rad) {

  // Get views
  View3D<double> uCF = state->u_cf();
  View3D<double> uPF = state->u_pf();
  View3D<double> uAF = state->u_af();

  // Grid parameters
  const int nX = grid.get_n_elements();
  const int ilo = grid.get_ilo();
  const int ihi = grid.get_ihi();
  const int modal_size = nX * order;

  // Generate filename
  constexpr int max_digits = 6;
  const auto& problem_name =
      pin->param()->get_ref<std::string>("problem.problem");
  std::string filename = generate_filename(problem_name, i_write, max_digits);

  // Create HDF5 writer
  HDF5Writer writer(filename);

  // Create group structure
  writer.createGroup("/metadata");
  writer.createGroup("/metadata/build");
  writer.createGroup("/grid");
  writer.createGroup("/conserved");
  writer.createGroup("/auxiliary");
  writer.createGroup("/diagnostic");
  writer.createGroup("/parameters");

  // Define variable configuration
  std::map<std::string, VariableInfo> variables{
      // Grid variables
      {"grid/x",
       {.path = "/grid/x", .description = "Cell centers", .is_modal = false}},
      {"grid/dx",
       {.path = "/grid/dx", .description = "Cell widths", .is_modal = false}},
      {"grid/x_nodal",
       {.path = "/grid/x_nodal",
        .description = "Nodal coordinates",
        .is_modal = true}},

      // Conserved variables
      {"conserved/tau",
       {.path = "/conserved/tau",
        .description = "Mass density",
        .is_modal = true}},
      {"conserved/velocity",
       {.path = "/conserved/velocity",
        .description = "Velocity",
        .is_modal = true}},
      {"conserved/energy",
       {.path = "/conserved/energy",
        .description = "Internal energy",
        .is_modal = true}},

      // Auxiliary variables
      {"auxiliary/pressure",
       {.path = "/auxiliary/pressure",
        .description = "Pressure",
        .is_modal = true}},

      // Diagnostic variables
      {"diagnostic/limiter",
       {.path = "/diagnostic/limiter",
        .description = "Slope limiter values",
        .is_modal = false}}};

  // Add radiation variables if needed
  if (do_rad) {
    variables["conserved/rad_energy"] = {.path = "/conserved/rad_energy",
                                         .description = "Radiation energy",
                                         .is_modal = true};
    variables["conserved/rad_momentum"] = {.path = "/conserved/rad_momentum",
                                           .description = "Radiation momentum",
                                           .is_modal = true};
  }

  // Prepare data containers
  std::map<std::string, std::vector<double>> data_arrays;

  // Initialize arrays based on variable type
  for (const auto& [key, var_info] : variables) {
    const int size = var_info.is_modal ? modal_size : nX;
    data_arrays[key] = std::vector<double>(size);
  }

  // Fill data arrays
  for (int iX = ilo; iX <= ihi; iX++) {
    const int i_local = iX - ilo;

    // Cell-centered quantities (filled once per cell)
    data_arrays["grid/x"][i_local] = grid.get_centers(iX);
    data_arrays["grid/dx"][i_local] = grid.get_widths(iX);
    data_arrays["diagnostic/limiter"][i_local] = get_limited(SL, iX);

    // Modal quantities (filled for each mode in each cell)
    for (int k = 0; k < order; k++) {
      const int idx = i_local + (k * nX);

      data_arrays["grid/x_nodal"][idx] = grid.node_coordinate(iX, k);
      data_arrays["conserved/tau"][idx] = uCF(iX, k, 0);
      data_arrays["conserved/velocity"][idx] = uCF(iX, k, 1);
      data_arrays["conserved/energy"][idx] = uCF(iX, k, 2);
      data_arrays["auxiliary/pressure"][idx] = uAF(iX, k, 0);

      if (do_rad) {
        data_arrays["conserved/rad_energy"][idx] = uCF(iX, k, 3);
        data_arrays["conserved/rad_momentum"][idx] = uCF(iX, k, 4);
      }
    }
  }

  // metadata
  writer.write_scalar("/metadata/nx", nX, H5::PredType::NATIVE_INT);
  writer.write_scalar("/metadata/order", order, H5::PredType::NATIVE_INT);
  writer.write_scalar("/metadata/time", time, H5::PredType::NATIVE_DOUBLE);

  // build information
  writer.write_string("/metadata/build/git_hash", build_info::GIT_HASH);
  writer.write_string("/metadata/build/compiler", build_info::COMPILER);
  writer.write_string("/metadata/build/timestamp", build_info::BUILD_TIMESTAMP);
  writer.write_string("/metadata/build/arch", build_info::ARCH);
  writer.write_string("/metadata/build/os", build_info::OS);
  writer.write_string("/metadata/build/optimization", build_info::OPTIMIZATION);

  // Write all variable data
  for (const auto& [key, var_info] : variables) {
    if (key.find("rad_") != std::string::npos && !do_rad) {
      continue;
    }
    writer.write_vector(var_info.path, data_arrays[key], var_info.description);
  }

  // deal with params
  const auto keys = pin->param()->keys();
  // probably a more elegant loop pattern
  for (const auto& key : keys) {
    auto t = pin->param()->get_type(key);
    if (t == typeid(std::string)) {
      writer.write_string("parameters/" + key,
                          pin->param()->get<std::string>(key));
    } else if (t == typeid(int)) {
      writer.write_scalar("parameters/" + key, pin->param()->get<int>(key),
                          H5::PredType::NATIVE_INT);
    } else if (t == typeid(double)) {
      writer.write_scalar("parameters/" + key, pin->param()->get<double>(key),
                          H5::PredType::NATIVE_DOUBLE);
    } else {
      // If the type cannot be matched, write a default string.
      writer.write_string("parameters/" + key, "Null");
    }
  }
}

/**
 * Write Modal basis coefficients and mass matrix
 **/
void write_basis(ModalBasis* basis, const int ihi, const int nNodes,
                 const int order, const std::string& problem_name) {
  std::string fn = problem_name;
  fn.append("_basis");
  fn.append(".h5");

  const char* fn2 = fn.c_str();

  static constexpr int ilo = 1;

  // Calculate total size needed
  const size_t total_size = static_cast<size_t>(ihi) * (nNodes + 2) * order;

  // Use std::vector instead of raw pointer for automatic memory management
  std::vector<double> data(total_size);

  // Fill data using vector indexing instead of pointer arithmetic
  for (int iX = ilo; iX <= ihi; iX++) {
    for (int iN = 0; iN < nNodes + 2; iN++) {
      for (int k = 0; k < order; k++) {
        const size_t idx = (((iX - ilo) * (nNodes + 2) + iN) * order) + k;
        data[idx] = basis->get_phi(static_cast<int>(iX), static_cast<int>(iN),
                                   static_cast<int>(k));
      }
    }
  }

  // Create HDF5 file and dataset
  H5::H5File const file(fn2, H5F_ACC_TRUNC);
  std::array<hsize_t, 3> dimsf = {static_cast<hsize_t>(ihi),
                                  static_cast<hsize_t>(nNodes + 2),
                                  static_cast<hsize_t>(order)};
  H5::DataSpace dataspace(3, dimsf.data());
  H5::DataSet basisDataset =
      file.createDataSet("basis", H5::PredType::NATIVE_DOUBLE, dataspace);

  // Write to File
  basisDataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
}
