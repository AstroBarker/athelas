/**
 * File     :  ProblemIn.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the problem initialization
 * See: https://github.com/marzer/tomlplusplus
 *
 **/

#include "ProblemIn.hpp"
#include "Error.hpp"

ProblemIn::ProblemIn( const std::string fn ) {
  // toml++ wants a string_view
  std::string_view fn_in{fn};

  // Load ini
  toml::table in_table;
  try {
    in_table = toml::parse_file( fn_in );
  }
  catch ( const toml::parse_error &err ) {
    std::cerr << err << "\n";
    throw Error( " ! Issue reading input deck!" );
  }

  std::printf("TESTING WHEFSE %d\n", in_table["Problem"]["nX"].value<int>());

  CSimpleIniA ini;
  SI_Error rc = ini.LoadFile( fn.c_str( ) );

  // Check if everything is happy
  if ( rc < 0 ) {
    throw Error( "Problem reading input deck\n" );
  }
  assert( rc == SI_OK );

  /* Grab as std::optional<type> */

  // problem
  std::optional<std::string> pn   = in_table["Problem"]["problem"].value<std::string>();
  std::optional<bool> rest = in_table["Problem"]["restart"].value<bool>();
  std::optional<bool> rad = in_table["Problem"]["do_rad"].value<bool>();
  std::optional<std::string> geom   = in_table["Problem"]["geometry"].value<std::string>();
  std::optional<Real> tf = in_table["Problem"]["t_end"].value<Real>();
  std::optional<Real> x1 = in_table["Problem"]["xL"].value<Real>();
  std::optional<Real> x2 = in_table["Problem"]["xR"].value<Real>();
  std::optional<std::string> bc   = in_table["Problem"]["BC"].value<std::string>();
  std::optional<Real> cfl = in_table["Problem"]["CFL"].value<Real>();

  // fluid
  std::optional<std::string> pn   = in_table["Fluid"]["Basis"].value<std::string>();
  std::optional<int> nN   = in_table["Fluid"]["nNodes"].value<int>();
  std::optional<int> nX   = in_table["Fluid"]["nX"].value<int>();
  std::optional<int> nG   = in_table["Fluid"]["nG"].value<int>();
  std::optional<int> pO   = in_table["Fluid"]["pO"].value<int>();

  // time
  std::optional<int> tO   = in_table["Time"]["tOrder"].value<int>();
  std::optional<int> nS   = in_table["Time"]["nStages"].value<int>();
  tO = ini.GetValue( "Time", "tOrder" );
  nS = ini.GetValue( "Time", "nStages" );

  // limiters
  std::optional<Real> al   = in_table["Limiters"]["alpha"].value<Real>();
  std::optional<Real> slt   = in_table["Limiters"]["threshold"].value<Real>();
  std::optional<bool> tci_opt   = in_table["Limiters"]["tci_opt"].value<bool>();
  std::optional<Real> tci_val   = in_table["Limiters"]["tci_val"].value<Real>();
  std::optional<bool> characteristic   = in_table["Limiters"]["characteristic1"].value<bool>();
  characteristic = ini.GetValue( "Limiters", "characteristic" );

  ProblemName = pn;
  BC          = bc;
  Restart     = ( strcmp( rest, "true" ) == 0 ) ? true : false;
  do_rad      = ( strcmp( rad, "true" ) == 0 ) ? true : false;
  Geometry    = ( strcmp( geom, "spherical" ) == 0 ) ? geometry::Spherical
                                                     : geometry::Planar;
  Basis       = ( strcmp( basis, "Legendre" ) == 0 ) ? PolyBasis::Legendre
                                                     : PolyBasis::Taylor;
  xL          = std::atof( x1 );
  xR          = std::atof( x2 );
  t_end       = std::atof( tf );
  CFL         = std::atof( cfl );

  nNodes    = std::atoi( nN );
  nElements = std::atoi( nX );
  nGhost    = std::atoi( nG );
  pOrder    = std::atoi( pO );
  tOrder    = std::atoi( tO );
  nStages   = std::atoi( nS );

  alpha          = std::atof( al );
  SL_Threshold   = std::atof( slt );
  TCI_Option     = ( strcmp( tci_opt, "true" ) == 0 ) ? true : false;
  TCI_Threshold  = std::atof( tci_val );
  Characteristic = ( strcmp( characteristic, "true" ) == 0 ) ? true : false;
}
