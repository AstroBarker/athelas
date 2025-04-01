/**
 * File     :  problem_in.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the problem initialization
 * See: https://github.com/marzer/tomlplusplus
 *
 **/

#include "problem_in.hpp"
#include "error.hpp"
#include "utilities.hpp"

ProblemIn::ProblemIn( const std::string fn ) {
  // toml++ wants a string_view
  std::string_view fn_in{ fn };

  // Load ini
  try {
    in_table = toml::parse_file( fn_in );
  } catch ( const toml::parse_error &err ) {
    std::cerr << err << "\n";
    THROW_ATHELAS_ERROR( " ! Issue reading input deck!" );
  }

  std::printf( " ~ Loading Input Deck ...\n" );

  /* Grab as std::optional<type> */

  // problem
  std::optional<std::string> pn =
      in_table["problem"]["problem"].value<std::string>( );
  std::optional<bool> rest = in_table["problem"]["restart"].value<bool>( );
  std::optional<bool> rad  = in_table["problem"]["do_rad"].value<bool>( );
  std::optional<std::string> geom =
      in_table["problem"]["geometry"].value<std::string>( );
  std::optional<Real> tf = in_table["problem"]["t_end"].value<Real>( );
  std::optional<Real> x1 = in_table["problem"]["xl"].value<Real>( );
  std::optional<Real> x2 = in_table["problem"]["xr"].value<Real>( );
  std::optional<std::string> bc =
      in_table["problem"]["bc"].value<std::string>( );
  std::optional<Real> cfl = in_table["problem"]["cfl"].value<Real>( );

  // output
  nlim         = in_table["output"]["nlim"].value_or( -1 );
  ncycle_out   = in_table["output"]["ncycle_out"].value_or( 1 );
  dt_hdf5      = in_table["output"]["dt_hdf5"].value_or( tf.value( ) / 100.0 );
  dt_init_frac = in_table["output"]["dt_init_frac"].value_or( 2.0 );
  if ( dt_init_frac <= 1.0 ) {
    THROW_ATHELAS_ERROR( "dt_init_frac must be strictly > 1.0\n" );
  }

  // fluid
  std::optional<std::string> basis =
      in_table["fluid"]["basis"].value<std::string>( );
  std::optional<int> nN = in_table["fluid"]["nnodes"].value<int>( );
  std::optional<int> nX = in_table["fluid"]["nx"].value<int>( );
  std::optional<int> nG = in_table["fluid"]["ng"].value<int>( );
  std::optional<int> pO = in_table["fluid"]["porder"].value<int>( );

  // time
  std::optional<int> tO = in_table["time"]["torder"].value<int>( );
  std::optional<int> nS = in_table["time"]["nstages"].value<int>( );

  // limiters
  std::optional<bool> tci_opt = in_table["limiters"]["tci_opt"].value<bool>( );
  std::optional<Real> tci_val = in_table["limiters"]["tci_val"].value<Real>( );
  std::optional<bool> characteristic =
      in_table["limiters"]["characteristic"].value<bool>( );
  std::optional<Real> gamma1 = in_table["limiters"]["gamma_l"].value<Real>( );
  std::optional<Real> gamma2 = in_table["limiters"]["gamma_i"].value<Real>( );
  std::optional<Real> gamma3 = in_table["limiters"]["gamma_r"].value<Real>( );
  std::optional<Real> wenor  = in_table["limiters"]["weno_r"].value<Real>( );

  if ( pn ) {
    problem_name = pn.value( );
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: problem not supplied in input deck." );
  }
  // Validity of problem_name checked in initialization.

  if ( bc ) {
    BC = utilities::to_lower( bc.value( ) );
    std::cout << BC << std::endl;
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: boundary condition not supplied in "
        "input deck." );
  }
  if ( BC != "homogenous" && BC != "reflecting" && BC != "shockless_noh" &&
       BC != "periodic" ) {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: Bad boundary condition choice. Choose: \n"
        " - homogenous \n"
        " - reflecting \n"
        " - periodic \n"
        " - shockless_noh" );
  }

  if ( geom ) {
    Geometry = ( utilities::to_lower( geom.value( ) ) == "spherical" )
                   ? geometry::Spherical
                   : geometry::Planar;
  } else {
    std::printf( "   - Defaulting to planar geometry!\n" );
    Geometry = geometry::Planar; // default
  }
  if ( basis ) {
    Basis = ( utilities::to_lower( basis.value( ) ) == "legendre" )
                ? PolyBasis::Legendre
                : PolyBasis::Taylor;
  } else {
    Basis = PolyBasis::Legendre;
    std::printf( "   - Defaulting to Legendre polynomial basis!\n" );
  }

  if ( x1 ) {
    xL = x1.value( );
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: xL not supplied in input deck." );
  }
  if ( x2 ) {
    xR = x2.value( );
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: xR not supplied in input deck." );
  }
  if ( x1 >= x2 ) THROW_ATHELAS_ERROR( " ! Initialization Error: x1 >= xz2" );

  if ( tf ) {
    t_end = tf.value( );
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: t_end not supplied in input deck." );
  }
  if ( tf <= 0.0 ) THROW_ATHELAS_ERROR( " ! Initialization Error: tf <= 0.0" );

  if ( nX ) {
    nElements = nX.value( );
  } else {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: nX not supplied in innput deck." );
  }

  // many defaults not mentioned below...
  CFL     = cfl.value_or( 0.5 );
  Restart = rest.value_or( false );
  do_rad  = rad.value_or( false );

  nGhost  = nG.value_or( 1 );
  pOrder  = pO.value_or( 1 );
  nNodes  = nN.value_or( 1 );
  tOrder  = tO.value_or( 1 );
  nStages = nS.value_or( 1 );

  TCI_Option     = tci_opt.value_or( false );
  TCI_Threshold  = tci_val.value_or( 0.1 );
  Characteristic = characteristic.value_or( false );
  gamma_l        = gamma1.value_or( 0.005 );
  gamma_i        = gamma2.value_or( 0.990 );
  gamma_r        = gamma3.value_or( 0.005 );

  // varous checks
  if ( CFL <= 0.0 ) THROW_ATHELAS_ERROR( " ! Initialization : CFL <= 0.0!" );
  if ( nGhost <= 0 ) THROW_ATHELAS_ERROR( " ! Initialization : nGhost <= 0!" );
  if ( pOrder <= 0 ) THROW_ATHELAS_ERROR( " ! Initialization : pOrder <= 0!" );
  if ( nNodes <= 0 ) THROW_ATHELAS_ERROR( " ! Initialization : nNodes <= 0!" );
  if ( tOrder <= 0 ) THROW_ATHELAS_ERROR( " ! Initialization : tOrder <= 0!" );
  if ( nStages <= 0 )
    THROW_ATHELAS_ERROR( " ! Initialization : nStages <= 0!" );
  if ( TCI_Threshold <= 0.0 )
    THROW_ATHELAS_ERROR( " ! Initialization : TCI_Threshold <= 0.0!" );

  if ( ( gamma2 && !gamma1 ) || ( gamma2 && !gamma3 ) ) {
    gamma_i = gamma2.value( );
    gamma_l = ( 1.0 - gamma_i ) / 2.0;
    gamma_r = ( 1.0 - gamma_i ) / 2.0;
  }
  const Real sum_g = gamma_l + gamma_i + gamma_r;
  if ( std::fabs( sum_g - 1.0 ) > 1.0e-10 ) {
    std::fprintf( stderr, "{gamma}, sum gamma = { %.10f %.10f %.10f }, %.18e\n",
                  gamma_l, gamma_i, gamma_r, 1.0 - sum_g );
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: Linear weights must sum to unity." );
  }
  weno_r = wenor.value_or( 2.0 );

  std::printf( " ~ Configuration ... Complete\n\n" );
}
