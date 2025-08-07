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

#include "pgen/problem_in.hpp"
#include "timestepper/tableau.hpp"
#include "utils/error.hpp"
#include "utils/utilities.hpp"

using namespace geometry;

// Provide access to the underlying params object
auto ProblemIn::param() -> Params* { return params_.get(); }

[[nodiscard]] auto ProblemIn::param() const -> Params* { return params_.get(); }

ProblemIn::ProblemIn(const std::string& fn) {
  // toml++ wants a string_view
  std::string_view const fn_in{fn};

  // Load ini
  try {
    config = toml::parse_file(fn_in);
  } catch (const toml::parse_error& err) {
    std::cerr << err << "\n";
    THROW_ATHELAS_ERROR(" ! Issue reading input deck!");
  }
  params_ = std::make_unique<Params>();

  std::println("# Loading Input Deck ...");

  // --- problem block ---
  if (!config["problem"].is_table()) {
    THROW_ATHELAS_ERROR("Input deck must have a [problem] block!");
  }

  std::optional<std::string> pname =
      config["problem"]["problem"].value<std::string>();
  if (!pname.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'problem' in [problem] block.");
  }
  params_->add("problem.problem", pname.value());

  std::optional<bool> restart = config["problem"]["restart"].value_or(false);
  params_->add("problem.restart", restart.value_or(false));

  std::optional<double> tf = config["problem"]["t_end"].value<double>();
  if (!tf.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'tf' in [problem] block.");
  }
  if (tf.value() <= 0.0) {
    THROW_ATHELAS_ERROR("tf must be > 0.0!");
  }
  params_->add("problem.tf", tf.value());

  const double nlim = config["problem"]["nlim"].value_or(-1);
  params_->add("problem.nlim", nlim);

  std::optional<double> xl = config["problem"]["xl"].value<double>();
  if (!xl.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'xl' in [problem] block.");
  }
  params_->add("problem.xl", xl.value());

  std::optional<double> xr = config["problem"]["xr"].value<double>();
  if (!xr.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'xr' in [problem] block.");
  }
  if (xr.value() <= xl.value()) {
    THROW_ATHELAS_ERROR("xr must be > xl!");
  }
  params_->add("problem.xr", xr.value());

  std::optional<int> nx = config["problem"]["nx"].value<int>();
  if (!nx.has_value()) {
    THROW_ATHELAS_ERROR("Missing nx in [problem] block!");
  }
  if (nx.value() <= 0) {
    THROW_ATHELAS_ERROR("nx must be > 0!");
  }
  params_->add("problem.nx", nx.value());

  std::optional<double> cfl = config["problem"]["cfl"].value<double>();
  // NOTE: It may be worthwhile to have cfl be registerd per physics.
  if (!cfl.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'cfl' in [problem] block.");
  }
  if (cfl.value() <= 0.0) {
    THROW_ATHELAS_ERROR("cfl must be > 0.0!");
  }
  params_->add("problem.cfl", cfl.value());

  std::optional<std::string> geom =
      config["problem"]["geometry"].value<std::string>();
  if (!geom.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'geom' in [problem] block.");
  }
  params_->add("problem.geometry", geom.value());
  if (geom.value() == "planar") {
    params_->add("problem.geometry_model", Geometry::Planar);
  }
  if (geom.value() == "spherical") {
    params_->add("problem.geometry_model", Geometry::Spherical);
  }

  // --- hande [problem.params] ---
  if (!config["problem"]["params"].is_table()) {
    THROW_ATHELAS_ERROR("No [params] block in [problem]!");
  }
  auto* pparams = config["problem"]["params"].as_table();
  for (auto&& [key, node] : *pparams) {
    // There must be a better way to do this...
    if (auto val = node.value<int>()) {
      std::string this_key = std::string{key};
      std::string out      = "problem.params." + this_key;
      params_->add(out, val.value());
    }
    if (auto val = node.value<double>()) {
      std::string this_key = std::string{key};
      std::string out      = "problem.params." + this_key;
      params_->add(out, val.value());
    }
    if (auto val = node.value<bool>()) {
      std::string this_key = std::string{key};
      std::string out      = "problem.params." + this_key;
      params_->add(out, val.value());
    }
    if (auto val = node.value<std::string>()) {
      std::string this_key = std::string{key};
      std::string out      = "problem.params." + this_key;
      params_->add(out, val.value());
    }
  }

  // --- physics block ---
  if (!config["physics"].is_table()) {
    THROW_ATHELAS_ERROR("Input deck must have a [physics] block!");
  }
  std::optional<bool> rad = config["physics"]["radiation"].value<bool>();
  if (!rad) {
    THROW_ATHELAS_ERROR("Missing or invalid 'radiation' in [physics] block.");
  }
  params_->add("physics.rad_active", rad.value());

  std::optional<bool> grav = config["physics"]["gravity"].value<bool>();
  if (!grav) {
    THROW_ATHELAS_ERROR("Missing or invalid 'gravity' in [physics] block.");
  }
  params_->add("physics.gravity_active", grav.value());

  // --- fluid block ---
  if (!config["fluid"].is_table()) {
    THROW_ATHELAS_ERROR("[fluid] block must be provided!");
  }

  std::optional<int> porder = config["fluid"]["porder"].value<int>();
  if (!porder) {
    THROW_ATHELAS_ERROR("fluid enabled but 'porder' missing in [fluid] block!");
  }
  params_->add("fluid.porder", porder.value());

  std::optional<int> nnodes = config["fluid"]["nnodes"].value<int>();
  if (!nnodes) {
    THROW_ATHELAS_ERROR("fluid enabled but 'nnodes' missing in [fluid] block!");
  }
  params_->add("fluid.nnodes", nnodes.value());

  if (!config["fluid"]["limiter"].is_table()) {
    WARNING_ATHELAS("No [limiter] block in [fluid] - defaulting to minmod with "
                    "standard values!");
  }

  std::optional<bool> limit_fluid =
      config["fluid"]["limiter"]["do_limiter"].value_or(true);
  params_->add("fluid.limiter.enabled", limit_fluid.value());

  std::optional<std::string> fluid_limiter =
      config["fluid"]["limiter"]["type"].value_or("minmod");
  params_->add("fluid.limiter.type", fluid_limiter.value());

  if (limit_fluid.value() && fluid_limiter.value() == "minmod") {
    const double b_tvd = config["fluid"]["limiter"]["b_tvd"].value_or(1.0);
    params_->add("fluid.limiter.b_tvd", b_tvd);
    const double m_tvb = config["fluid"]["limiter"]["m_tvb"].value_or(0.0);
    params_->add("fluid.limiter.m_tvb", m_tvb);
  }
  if (limit_fluid.value() && fluid_limiter.value() == "weno") {
    std::optional<double> gamma_i =
        config["fluid"]["limiter"]["gamma_i"].value<double>();
    std::optional<double> gamma_l =
        config["fluid"]["limiter"]["gamma_l"].value<double>();
    std::optional<double> gamma_r =
        config["fluid"]["limiter"]["gamma_r"].value<double>();
    if ((gamma_i && !gamma_l) || (gamma_i && !gamma_r)) {
      params_->add("fluid.limiter.gamma_i", gamma_i.value());
      params_->add("fluid.limiter.gamma_l", (1.0 - gamma_i.value()) / 2.0);
      params_->add("fluid.limiter.gamma_r", (1.0 - gamma_i.value()) / 2.0);
    } else if (gamma_i && gamma_r && gamma_l) {
      params_->add("fluid.limiter.gamma_i", gamma_i.value());
      params_->add("fluid.limiter.gamma_r", gamma_r.value());
      params_->add("fluid.limiter.gamma_l", gamma_l.value());
    } else {
      THROW_ATHELAS_ERROR("Error parsing weno gammas in [fluid] block: provide "
                          "only gamma_i, or all gamma_l, gamma_i, gamma_r!");
    }
    const double sum_g = params_->get<double>("fluid.limiter.gamma_i") +
                         params_->get<double>("fluid.limiter.gamma_l") +
                         params_->get<double>("fluid.limiter.gamma_r");
    if (std::abs(sum_g - 1.0) > 1.0e-10) {
      THROW_ATHELAS_ERROR(
          " ! Initialization Error: Linear WENO weights must sum to unity.");
    }
    const double weno_r = config["fluid"]["limiter"]["weno_r"].value_or(2.0);
    if (weno_r <= 0.0) {
      THROW_ATHELAS_ERROR(
          "[fluid] block: WENO limiter weno_r must be positive!");
    }
    params_->add("fluid.limiter.weno_r", weno_r);
  }

  // tci
  const bool do_tci = config["fluid"]["limiter"]["tci_opt"].value_or(false);
  params_->add("fluid.limiter.tci_enabled", do_tci);
  if (do_tci) {
    std::optional<double> tci_val =
        config["fluid"]["limiter"]["tci_val"].value<double>();
    if (!tci_val.has_value()) {
      THROW_ATHELAS_ERROR(
          "[fluid] block: TCI requested but no tci_val provided!");
    }
    params_->add("fluid.limiter.tci_val", tci_val.value());
  } else {
    params_->add("fluid.limiter.tci_val", 0.0);
  }

  // characteristic limiting
  const bool characteristic =
      config["fluid"]["limiter"]["characteristic"].value_or(false);
  params_->add("fluid.limiter.characteristic", characteristic);

  // fluid bc
  std::optional<std::string> fluid_bc_i =
      config["bc"]["fluid"]["bc_i"].value<std::string>();
  std::optional<std::string> fluid_bc_o =
      config["bc"]["fluid"]["bc_o"].value<std::string>();

  // --- fluid bc ---
  if (fluid_bc_i.has_value()) {
    params_->add("fluid.bc.i", utilities::to_lower(fluid_bc_i.value()));
  } else {
    THROW_ATHELAS_ERROR("Inner fluid boundary condition not supplied "
                        "in input deck.");
  }
  if (fluid_bc_o.has_value()) {
    params_->add("fluid.bc.o", utilities::to_lower(fluid_bc_o.value()));
  } else {
    THROW_ATHELAS_ERROR("Outer fluid boundary condition not supplied "
                        "in input deck.");
  }
  check_bc(params_->get<std::string>("fluid.bc.i"));
  check_bc(params_->get<std::string>("fluid.bc.i"));

  // handle dirichlet..
  std::array<double, 3> fluid_i_dirichlet_values = {0.0, 0.0, 0.0};
  std::array<double, 3> fluid_o_dirichlet_values = {0.0, 0.0, 0.0};

  if (fluid_bc_i == "dirichlet") {
    const auto& node = config["bc"]["fluid"]["dirichlet_values_i"];
    if (node && node.is_array()) {
      const auto* array = node.as_array();
      read_toml_array(array, fluid_i_dirichlet_values);
    } else {
      THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read fluid "
                          "dirichlet_values_i as array.");
    }
  }

  if (fluid_bc_o == "dirichlet") {
    const auto& node = config["bc"]["fluid"]["dirichlet_values_o"];
    if (node && node.is_array()) {
      const auto* array = node.as_array();
      read_toml_array(array, fluid_o_dirichlet_values);
    } else {
      THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read fluid "
                          "dirichlet_values_o as array.");
    }
  }
  params_->add("fluid.bc.i.dirichlet_values", fluid_i_dirichlet_values);
  params_->add("fluid.bc.o.dirichlet_values", fluid_o_dirichlet_values);
  // fluid block

  // --- radiation block ---
  // I suspect much of this should really go into
  // the individual packages.
  if (rad.value()) {
    if (!config["radiation"].is_table()) {
      THROW_ATHELAS_ERROR(
          "Radiation is active but radiation block is missing!");
    }

    std::optional<int> porder = config["radiation"]["porder"].value<int>();
    if (!porder) {
      THROW_ATHELAS_ERROR(
          "radiation enabled but 'porder' missing in [radiation] block!");
    }
    params_->add("radiation.porder", porder.value());

    std::optional<int> nnodes = config["radiation"]["nnodes"].value<int>();
    if (!nnodes) {
      THROW_ATHELAS_ERROR(
          "radiation enabled but 'nnodes' missing in [radiation] block!");
    }
    params_->add("radiation.nnodes", nnodes.value());

    if (!config["radiation"]["limiter"].is_table()) {
      WARNING_ATHELAS("No [limiter] block in [radiation] - defaulting to "
                      "minmod with standard values!");
    }

    std::optional<bool> limit_rad =
        config["radiation"]["limiter"]["do_limiter"].value_or(true);
    params_->add("radiation.limiter.enabled", limit_rad.value());

    std::optional<std::string> rad_limiter =
        config["radiation"]["limiter"]["type"].value_or("minmod");
    params_->add("radiation.limiter.type", rad_limiter.value());

    if (limit_rad.value() && rad_limiter.value() == "minmod") {
      const double b_tvd =
          config["radiation"]["limiter"]["b_tvd"].value_or(1.0);
      params_->add("radiation.limiter.b_tvd", b_tvd);
      const double m_tvb =
          config["radiation"]["limiter"]["m_tvb"].value_or(0.0);
      params_->add("radiation.limiter.m_tvb", m_tvb);
    }
    if (limit_rad.value() && rad_limiter.value() == "weno") {
      std::optional<double> gamma_i =
          config["radiation"]["limiter"]["gamma_i"].value<double>();
      std::optional<double> gamma_l =
          config["radiation"]["limiter"]["gamma_l"].value<double>();
      std::optional<double> gamma_r =
          config["radiation"]["limiter"]["gamma_r"].value<double>();
      if ((gamma_i && !gamma_l) || (gamma_i && !gamma_r)) {
        params_->add("radiation.limiter.gamma_i", gamma_i.value());
        params_->add("radiation.limiter.gamma_l",
                     (1.0 - gamma_i.value()) / 2.0);
        params_->add("radiation.limiter.gamma_r",
                     (1.0 - gamma_i.value()) / 2.0);
      } else if (gamma_i && gamma_r && gamma_l) {
        params_->add("radiation.limiter.gamma_i", gamma_i.value());
        params_->add("radiation.limiter.gamma_r", gamma_r.value());
        params_->add("radiation.limiter.gamma_l", gamma_l.value());
      } else {
        THROW_ATHELAS_ERROR(
            "Error parsing weno gammas in [radiation] block: provide only "
            "gamma_i, or all gamma_l, gamma_i, gamma_r!");
      }
      const double sum_g = params_->get<double>("radiation.limiter.gamma_i") +
                           params_->get<double>("radiation.limiter.gamma_l") +
                           params_->get<double>("radiation.limiter.gamma_r");
      if (std::abs(sum_g - 1.0) > 1.0e-10) {
        THROW_ATHELAS_ERROR(
            " ! Initialization Error: Linear WENO weights must sum to unity.");
      }
      const double weno_r =
          config["radiation"]["limiter"]["weno_r"].value_or(2.0);
      if (weno_r <= 0.0) {
        THROW_ATHELAS_ERROR(
            "[radiation] block: WENO limiter weno_r must be positive!");
      }
      params_->add("radiation.limiter.weno_r", weno_r);
    }

    // tci
    const bool do_tci =
        config["radiation"]["limiter"]["tci_opt"].value_or(false);
    params_->add("radiation.limiter.tci_enabled", do_tci);
    if (do_tci) {
      std::optional<double> tci_val =
          config["radiation"]["limiter"]["tci_val"].value<double>();
      if (!tci_val.has_value()) {
        THROW_ATHELAS_ERROR(
            "[radiation] block: TCI requested but no tci_val provided!");
      }
      params_->add("radiation.limiter.tci_val", tci_val.value());
    } else {
      params_->add("radiation.limiter.tci_val", 0.0);
    }

    // characteristic limiting
    const bool characteristic =
        config["radiation"]["limiter"]["characteristic"].value_or(false);
    params_->add("radiation.limiter.characteristic", characteristic);

    // --- radiation bc ---
    std::optional<std::string> rad_bc_i =
        config["bc"]["radiation"]["bc_i"].value<std::string>();
    std::optional<std::string> rad_bc_o =
        config["bc"]["radiation"]["bc_o"].value<std::string>();

    if (rad_bc_i.has_value()) {
      params_->add("radiation.bc.i", utilities::to_lower(rad_bc_i.value()));
    } else {
      THROW_ATHELAS_ERROR("Inner radiation boundary condition not supplied "
                          "in input deck but radiation is enabled1");
    }
    if (rad_bc_o.has_value()) {
      params_->add("radiation.bc.o", utilities::to_lower(rad_bc_o.value()));
    } else {
      THROW_ATHELAS_ERROR("Outer radiation boundary condition not supplied "
                          "in input deck but radiation is enabled!");
    }
    check_bc(params_->get<std::string>("radiation.bc.i"));
    check_bc(params_->get<std::string>("radiation.bc.i"));

    // handle dirichlet..
    std::array<double, 2> rad_i_dirichlet_values = {0.0, 0.0};
    std::array<double, 2> rad_o_dirichlet_values = {0.0, 0.0};

    if (rad_bc_i == "dirichlet" || rad_bc_i == "marshak") {
      const auto& node = config["bc"]["radiation"]["dirichlet_values_i"];
      if (node && node.is_array()) {
        const auto* array = node.as_array();
        read_toml_array(array, rad_i_dirichlet_values);
      } else {
        THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read radiation "
                            "dirichlet_values_i as array.");
      }
    }

    if (rad_bc_o == "dirichlet") {
      const auto& node = config["bc"]["radiation"]["dirichlet_values_o"];
      if (node && node.is_array()) {
        const auto* array = node.as_array();
        read_toml_array(array, rad_o_dirichlet_values);
      } else {
        THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read radiation "
                            "dirichlet_values_o as array.");
      }
    }
    params_->add("radiation.bc.i.dirichlet_values", rad_i_dirichlet_values);
    params_->add("radiatio.bc.o.dirichlet_values", rad_o_dirichlet_values);
  } // radiation block

  // gravity block --
  if (grav.value()) {
    if (!config["gravity"].is_table()) {
      THROW_ATHELAS_ERROR(
          "Gravity is enabled but no [gravity] block exists in input deck!");
    }
    const double gval = config["gravity"]["gval"].value_or(1.0);
    params_->add("gravity.gval", gval); // Always present
    const std::string gmodel = config["gravity"]["model"].value_or("constant");
    params_->add("gravity.model", (utilities::to_lower(gmodel) == "spherical")
                                      ? GravityModel::Spherical
                                      : GravityModel::Constant);
    if (params_->get<GravityModel>("gravity.model") == GravityModel::Constant &&
        gval <= 0.0) {
      THROW_ATHELAS_ERROR(
          "Constant gravitational potential requested but g <= 0.0!");
    }
  } // gravity block

  // --- output ---
  // In principle everything below can be defaulted, but
  // I still require the block present.
  if (!config["output"].is_table()) {
    THROW_ATHELAS_ERROR("No [output] block provided!");
  }
  const int ncycle_out = config["output"]["ncycle_out"].value_or(1);
  const double dt_hdf5 =
      config["output"]["dt_hdf5"].value_or(tf.value() / 100.0);
  const double dt_init_frac  = config["output"]["dt_init_frac"].value_or(1.05);
  const double initial_dt    = config["output"]["initial_dt"].value_or(1.0e-16);
  const bool history_enabled = config["output"]["history"].is_table();
  const std::string hist_fn =
      config["output"]["history"]["fn"].value_or("athelas.hst");
  const double hist_dt =
      config["output"]["history"]["dt"].value_or(dt_hdf5 / 10);
  if (initial_dt <= 0.0) {
    THROW_ATHELAS_ERROR("initial_dt must be strictly > 0.0\n");
  }
  if (dt_init_frac <= 1.0) {
    THROW_ATHELAS_ERROR("dt_init_frac must be strictly > 1.0\n");
  }
  if (dt_hdf5 <= 0.0) {
    THROW_ATHELAS_ERROR("dt_hdf5 must be strictly > 0.0\n");
  }
  if (hist_dt <= 0.0) {
    THROW_ATHELAS_ERROR("hist_dt must be strictly > 0.0\n");
  }
  params_->add("output.ncycle_out", ncycle_out);
  params_->add("output.dt_hdf5", dt_hdf5);
  params_->add("output.dt_init_frac", dt_init_frac);
  params_->add("output.initial_dt", initial_dt);
  params_->add("output.history_enabled", history_enabled);
  params_->add("output.hist_fn", hist_fn);
  params_->add("output.hist_dt", hist_dt);

  // --- time ---
  if (!config["time"].is_table()) {
    THROW_ATHELAS_ERROR("No [time] block provided!");
  }
  std::optional<std::string> integrator =
      config["time"]["integrator"].value<std::string>();
  if (integrator.has_value()) {
    const MethodID method_id =
        string_to_id(utilities::to_lower(integrator.value()));
    params_->add("time.integrator", method_id);
    params_->add("time.integrator_string", integrator.value()); // for IO
  } else {
    THROW_ATHELAS_ERROR("You must list an integrator in the input deck!");
  }

  // --- eos ---
  if (!config["eos"].is_table()) {
    THROW_ATHELAS_ERROR("No [eos] block provided!");
  }
  std::optional<std::string> eos_type =
      config["eos"]["type"].value<std::string>();
  if (!eos_type.has_value()) {
    THROW_ATHELAS_ERROR("'type' not provided in [eos] block!");
  }
  params_->add("eos.type", eos_type.value());
  params_->add("eos.gamma", config["eos"]["gamma"].value_or(1.4));

  // --- opac ---
  if (rad.value()) {
    if (!config["opacity"].is_table()) {
      THROW_ATHELAS_ERROR("Radiation abled but no [opac] block provided!");
    }
    std::optional<std::string> opac_type =
        config["opacity"]["type"].value<std::string>();
    if (!opac_type.has_value()) {
      THROW_ATHELAS_ERROR("'type' not provided in [opac] block!");
    }
    params_->add("opac.type", opac_type.value());

    if (opac_type.value() == "constant") {
      std::optional<double> kr = config["opacity"]["kR"].value<double>();
      std::optional<double> kp = config["opacity"]["kP"].value<double>();
      if (!kr.has_value() || !kp.has_value()) {
        THROW_ATHELAS_ERROR(
            "Constant opacity must specify mean opacities kR and kP!");
      }
      params_->add("opac.kR", kr.value());
      params_->add("opac.kP", kp.value());
    }
    if (opac_type.value() == "powerlaw_rho") {
      std::optional<double> kr = config["opacity"]["kR"].value<double>();
      std::optional<double> kp = config["opacity"]["kP"].value<double>();
      std::optional<double> exp =
          config["opacity"]["exp"].value<double>(); // exponent
      if (!kr.has_value() || !kp.has_value() || !exp.has_value()) {
        THROW_ATHELAS_ERROR("Powerlaw rho opacity must specify mean opacities "
                            "kR and kP and an exponent exp!");
      }
      params_->add("opac.kR", kr.value());
      params_->add("opac.kP", kp.value());
      params_->add("opac.exp", exp.value());
    }
  }

  std::println("# Configuration ... Complete\n");
}

auto check_bc(std::string bc) -> bool {
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
