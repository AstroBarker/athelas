/**
 * @file problem_in.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Class for loading input deck
 *
 * @details Loads input deck in TOML format.
 *          See: https://github.com/marzer/tomlplusplus
 *
 *  Horrible stuff here.
 */

#include <limits>

#include "error.hpp"
#include "problem_in.hpp"
#include "utilities.hpp"

ProblemIn::ProblemIn(const std::string& fn) {
  // toml++ wants a string_view
  std::string_view const fn_in{fn};

  // Load ini
  try {
    in_table = toml::parse_file(fn_in);
  } catch (const toml::parse_error& err) {
    std::cerr << err << "\n";
    THROW_ATHELAS_ERROR(" ! Issue reading input deck!");
  }

  std::println("# Loading Input Deck ...");

  // --- problem block ---
  if (!in_table["problem"].is_table()) {
    THROW_ATHELAS_ERROR("Input deck must have a [problem] block!");
  }

  std::optional<std::string> pname = in_table["problem"]["problem"].value<std::string>();
  if (!pname.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'problem' in [problem] block.");
  }
  params_.add("problem.problem_name", pname.value());

  std::optional<bool> restart = in_table["problem"]["restart"].value_or(false);
  params_.add("problem.restart", restart.value_or(false));

  std::optional<double> tf = in_table["problem"]["t_end"].value<double>();
  if (!tf.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'tf' in [problem] block.");
  }
  params_.add("problem.tf", tf.value());

  std::optional<double> xl = in_table["problem"]["xl"].value<double>();
  if (!xl.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'xl' in [problem] block.");
  }
  params_.add("problem.xl", xl.value());

  std::optional<double> xr = in_table["problem"]["xr"].value<double>();
  if (!xr.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'xr' in [problem] block.");
  }
  params_.add("problem.xr", xr.value());

  std::optional<double> cfl = in_table["problem"]["cfl"].value<double>();
  // NOTE: It may be worthwhile to have cfl be registerd per physics.
  if (!cfl.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'cfl' in [problem] block.");
  }
  params_.add("problem.cfl", cfl.value());

  std::optional<std::string> geom =
      in_table["problem"]["geometry"].value<std::string>();
  if (!geom.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'geom' in [problem] block.");
  }
  params_.add("problem.geometry", geom.value());

  // --- physics block ---
  if (!in_table["physics"].is_table()) {
    THROW_ATHELAS_ERROR("Input deck must have a [physics] block!");
  }
  std::optional<bool> rad = in_table["problem"]["radiation"].value<bool>();
  if (!rad) {
    THROW_ATHELAS_ERROR("Missing or invalid 'radiation' in [physics] block.");
  }
  params_.add("physics.rad_active", rad.value());

  std::optional<bool> grav = in_table["problem"]["gravity"].value<bool>();
  if (!grav) {
    THROW_ATHELAS_ERROR("Missing or invalid 'gravity' in [physics] block.");
  }
  params_.add("physics.gravity_active", rad.value());

  // parse radiation
  if (rad.value())

  // Is this a good pattern?
  do_gravity = in_table["problem"]["do_gravity"].value_or(false);
  if (do_gravity) {
    if (!in_table["gravity"].is_table()) {
      THROW_ATHELAS_ERROR(
          "Gravity is enabled but not gravity block exists in input deck!");
    } else {
      gval = in_table["gravity"]["gval"].value_or(0.0);
      const std::string gmodel =
          in_table["gravity"]["model"].value_or("constant");
      grav_model = (utilities::to_lower(gmodel) == "spherical")
                       ? GravityModel::Spherical
                       : GravityModel::Constant;
      if (grav_model == GravityModel::Constant && gval <= 0.0) {
        THROW_ATHELAS_ERROR(
            "Constant gravitational potential requested but g <= 0.0!");
      }
    }
  }

  // output
  nlim            = in_table["output"]["nlim"].value_or(-1);
  ncycle_out      = in_table["output"]["ncycle_out"].value_or(1);
  dt_hdf5         = in_table["output"]["dt_hdf5"].value_or(tf.value() / 100.0);
  dt_init_frac    = in_table["output"]["dt_init_frac"].value_or(2.0);
  history_enabled = in_table["output"]["history"].is_table();
  hist_fn         = in_table["output"]["history"]["fn"].value_or("athelas.hst");
  hist_dt         = in_table["output"]["history"]["dt"].value_or(dt_hdf5 / 10);
  if (dt_init_frac <= 1.0) {
    THROW_ATHELAS_ERROR("dt_init_frac must be strictly > 1.0\n");
  }
  if (dt_hdf5 <= 0.0) {
    THROW_ATHELAS_ERROR("dt_hdf5 must be strictly > 0.0\n");
  }
  if (hist_dt <= 0.0) {
    THROW_ATHELAS_ERROR("hist_dt must be strictly > 0.0\n");
  }

  // fluid
  std::optional<std::string> basis_ =
      in_table["fluid"]["basis"].value<std::string>();
  std::optional<int> nN = in_table["fluid"]["nnodes"].value<int>();
  std::optional<int> nX = in_table["fluid"]["nx"].value<int>();
  std::optional<int> nG = in_table["fluid"]["ng"].value<int>();
  std::optional<int> pO = in_table["fluid"]["porder"].value<int>();

  // rad
  do_rad = rad.value_or(false);

  // time
  std::optional<int> tO = in_table["time"]["torder"].value<int>();
  std::optional<int> nS = in_table["time"]["nstages"].value<int>();
  std::optional<std::string> integrator_ =
      in_table["time"]["integrator"].value<std::string>();
  if (integrator_) {
    integrator = utilities::to_lower(integrator_.value());
    method_id  = string_to_id(integrator);
  } else {
    THROW_ATHELAS_ERROR("You must list an integrator in the input deck!");
  }

  // eos
  eos_type = in_table["eos"]["type"].value_or("ideal");

  // opac
  opac_type = in_table["opacity"]["type"].value_or("constant");
  // not storing opac args

  // limiters
  do_limiter = in_table["limiters"]["do_limiter"].value_or(true);
  std::optional<bool> tci_opt = in_table["limiters"]["tci_opt"].value<bool>();
  std::optional<double> tci_val =
      in_table["limiters"]["tci_val"].value<double>();
  std::optional<bool> characteristic =
      in_table["limiters"]["characteristic"].value<bool>();
  std::optional<double> gamma1 =
      in_table["limiters"]["gamma_l"].value<double>();
  std::optional<double> gamma2 =
      in_table["limiters"]["gamma_i"].value<double>();
  std::optional<double> gamma3 =
      in_table["limiters"]["gamma_r"].value<double>();
  std::optional<double> wenor = in_table["limiters"]["weno_r"].value<double>();
  b_tvd                       = in_table["limiters"]["b_tvd"].value_or(1.0);
  m_tvb                       = in_table["limiters"]["m_tvb"].value_or(1.0);
  limiter_type                = in_table["limiters"]["type"].value_or("minmod");
  // if ( b_tvd < 1.0 || b_tvd > 2.0 ) {
  //   THROW_ATHELAS_ERROR( "b_tvd must be in [1.0, 2.0]." );
  // }
  if (utilities::to_lower(limiter_type) != "minmod" &&
      utilities::to_lower(limiter_type) != "weno") {
    THROW_ATHELAS_ERROR("Please choose a valid limiter! Current options: \n"
                        " - weno \n"
                        " - minmod");
  }

  std::optional<std::string> fluid_bc_i_ =
      in_table["bc"]["fluid"]["bc_i"].value<std::string>();
  std::optional<std::string> fluid_bc_o_ =
      in_table["bc"]["fluid"]["bc_o"].value<std::string>();
  std::optional<std::string> rad_bc_i_ =
      in_table["bc"]["rad"]["bc_i"].value<std::string>();
  std::optional<std::string> rad_bc_o_ =
      in_table["bc"]["rad"]["bc_o"].value<std::string>();

  if (pn) {
    problem_name = pn.value();
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: problem not supplied in input deck.");
  }
  // Validity of problem_name checked in initialization.

  // --- fluid bc ---
  if (fluid_bc_i_) {
    fluid_bc_i = utilities::to_lower(fluid_bc_i_.value());
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: inner fluid boundary condition not supplied "
        "in input deck.");
  }
  if (fluid_bc_o_) {
    fluid_bc_o = utilities::to_lower(fluid_bc_o_.value());
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: outer fluid boundary condition not supplied "
        "in input deck.");
  }
  check_bc(fluid_bc_i);
  check_bc(fluid_bc_o);

  // handle dirichlet..
  fluid_i_dirichlet_values = {0.0, 0.0, 0.0};
  fluid_o_dirichlet_values = {0.0, 0.0, 0.0};
  // --- testing ---
  auto* array = in_table["bc"]["fluid"]["dirichlet_values_i"].as_array();
  if (array && fluid_bc_i == "dirichlet") {
    read_toml_array(array, fluid_i_dirichlet_values);
  } else if (!array && fluid_bc_i == "dirichlet") {
    THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read fluid "
                        "dirichlet_values_i as array.");
  }
  array = in_table["bc"]["fluid"]["dirichlet_values_o"].as_array();
  if (array && fluid_bc_o == "dirichlet") {
    read_toml_array(array, fluid_o_dirichlet_values);
  } else if (!array && fluid_bc_o == "dirichlet") {
    THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read fluid "
                        "dirichlet_values_o as array.");
  }

  // --- rad bc ---
  rad_bc_i = "";
  rad_bc_o = "";
  if (rad_bc_i_) {
    rad_bc_i = utilities::to_lower(rad_bc_i_.value());
  } else if (!rad_bc_i_ && do_rad) {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: inner radiation boundary condition not "
        "supplied in input deck but radiation was enabled.");
  }
  if (rad_bc_o_) {
    rad_bc_o = utilities::to_lower(rad_bc_o_.value());
  } else if (!rad_bc_o_ && do_rad) {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: outer radiation boundary condition not "
        "supplied in input deck but radiation was enabled.");
  }
  if (do_rad) {
    check_bc(rad_bc_i);
    check_bc(rad_bc_o);
  }

  // handle dirichlet..
  rad_i_dirichlet_values = {0.0, 0.0};
  rad_o_dirichlet_values = {0.0, 0.0};
  const bool do_dirichlet_rad_i =
      (rad_bc_i == "dirichlet" || rad_bc_i == "marshak");
  const bool do_dirichlet_rad_o =
      (rad_bc_o == "dirichlet" || rad_bc_o == "marshak");
  array = in_table["bc"]["rad"]["dirichlet_values_i"].as_array();
  if (array && do_dirichlet_rad_i) {
    read_toml_array(array, rad_i_dirichlet_values);
  } else if (!array && do_dirichlet_rad_i) {
    THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read rad "
                        "dirichlet_values_i as array.");
  }
  array = in_table["bc"]["rad"]["dirichlet_values_o"].as_array();
  if (array && do_dirichlet_rad_o) {
    read_toml_array(array, rad_o_dirichlet_values);
  } else if (!array && do_dirichlet_rad_o) {
    THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read rad "
                        "dirichlet_values_o as array.");
  }

  if (geom) {
    Geometry = (utilities::to_lower(geom.value()) == "spherical")
                   ? geometry::Spherical
                   : geometry::Planar;
  } else {
    std::println("   - Defaulting to planar geometry!");
    Geometry = geometry::Planar; // default
  }
  if (basis_) {
    basis = (utilities::to_lower(basis_.value()) == "legendre")
                ? poly_basis::legendre
                : poly_basis::taylor;
  } else {
    basis = poly_basis::legendre;
    std::println("   - Defaulting to legendre polynomial basis!");
  }

  if (x1) {
    xL = x1.value();
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: xL not supplied in input deck.");
  }
  if (x2) {
    xR = x2.value();
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: xR not supplied in input deck.");
  }
  if (x1 >= x2) {
    THROW_ATHELAS_ERROR(" ! Initialization Error: x1 >= xz2");
  }

  if (tf) {
    t_end = tf.value();
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: t_end not supplied in input deck.");
  }
  if (tf <= 0.0) {
    THROW_ATHELAS_ERROR(" ! Initialization Error: tf <= 0.0");
  }

  if (nX) {
    nElements = nX.value();
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: nX not supplied in innput deck.");
  }

  // many defaults not mentioned below...
  CFL     = cfl.value_or(0.5);
  Restart = rest.value_or(false);

  nGhost  = nG.value_or(1);
  pOrder  = pO.value_or(1);
  nNodes  = nN.value_or(1);
  tOrder  = tO.value_or(1);
  nStages = nS.value_or(1);

  TCI_Option     = tci_opt.value_or(false);
  TCI_Threshold  = tci_val.value_or(0.1);
  Characteristic = characteristic.value_or(false);
  gamma_l        = gamma1.value_or(0.005);
  gamma_i        = gamma2.value_or(0.990);
  gamma_r        = gamma3.value_or(0.005);

  // various checks
  if (CFL <= 0.0) {
    THROW_ATHELAS_ERROR(" ! Initialization : CFL <= 0.0!");
  }
  if (nGhost <= 0) {
    THROW_ATHELAS_ERROR(" ! Initialization : nGhost <= 0!");
  }
  if (pOrder <= 0) {
    THROW_ATHELAS_ERROR(" ! Initialization : pOrder <= 0!");
  }
  if (nNodes <= 0) {
    THROW_ATHELAS_ERROR(" ! Initialization : nNodes <= 0!");
  }
  if (tOrder <= 0) {
    THROW_ATHELAS_ERROR(" ! Initialization : tOrder <= 0!");
  }
  if (nStages <= 0) {
    THROW_ATHELAS_ERROR(" ! Initialization : nStages <= 0!");
  }
  if (TCI_Threshold <= 0.0) {
    THROW_ATHELAS_ERROR(" ! Initialization : TCI_Threshold <= 0.0!");
  }

  if ((gamma2 && !gamma1) || (gamma2 && !gamma3)) {
    gamma_i = gamma2.value();
    gamma_l = (1.0 - gamma_i) / 2.0;
    gamma_r = (1.0 - gamma_i) / 2.0;
  }
  const double sum_g = gamma_l + gamma_i + gamma_r;
  if (std::fabs(sum_g - 1.0) > 1.0e-10) {
    std::fprintf(stderr, "{gamma}, sum gamma = { %.10f %.10f %.10f }, %.18e\n",
                 gamma_l, gamma_i, gamma_r, 1.0 - sum_g);
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: Linear weights must sum to unity.");
  }
  weno_r = wenor.value_or(2.0);

  std::println("# Configuration ... Complete\n");
}

auto check_bc(std::string& bc) -> bool {
  if (bc != "outflow" && bc != "reflecting" && bc != "dirichlet" &&
      bc != "periodic" && bc != "marshak") {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: Bad boundary condition choice. Choose: \n"
        " - outflow \n"
        " - reflecting \n"
        " - periodic \n"
        " - dirichlet");
  }
  return false; // should not reach
}
