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
#include "Utilities.hpp"

ProblemIn::ProblemIn( const std::string fn ) {
  // toml++ wants a string_view
  std::string_view fn_in{ fn };

  // Load ini
  try {
    in_table = toml::parse_file( fn_in );
  } catch ( const toml::parse_error &err ) {
    std::cerr << err << "\n";
    throw Error( " ! Issue reading input deck!" );
  }

  /* Grab as std::optional<type> */

  // problem
  std::optional<std::string> pn =
      in_table["Problem"]["problem"].value<std::string>( );
  std::optional<bool> rest = in_table["Problem"]["restart"].value<bool>( );
  std::optional<bool> rad  = in_table["Problem"]["do_rad"].value<bool>( );
  std::optional<std::string> geom =
      in_table["Problem"]["geometry"].value<std::string>( );
  std::optional<Real> tf = in_table["Problem"]["t_end"].value<Real>( );
  std::optional<Real> x1 = in_table["Problem"]["xL"].value<Real>( );
  std::optional<Real> x2 = in_table["Problem"]["xR"].value<Real>( );
  std::optional<std::string> bc =
      in_table["Problem"]["BC"].value<std::string>( );
  std::optional<Real> cfl = in_table["Problem"]["CFL"].value<Real>( );

  // fluid
  std::optional<std::string> basis =
      in_table["Fluid"]["Basis"].value<std::string>( );
  std::optional<int> nN = in_table["Fluid"]["nNodes"].value<int>( );
  std::optional<int> nX = in_table["Fluid"]["nX"].value<int>( );
  std::optional<int> nG = in_table["Fluid"]["nG"].value<int>( );
  std::optional<int> pO = in_table["Fluid"]["pOrder"].value<int>( );

  // time
  std::optional<int> tO = in_table["Time"]["tOrder"].value<int>( );
  std::optional<int> nS = in_table["Time"]["nStages"].value<int>( );

  // limiters
  std::optional<Real> al  = in_table["Limiters"]["alpha"].value<Real>( );
  std::optional<Real> slt = in_table["Limiters"]["Threshold"].value<Real>( );
  std::optional<bool> tci_opt = in_table["Limiters"]["TCI_opt"].value<bool>( );
  std::optional<Real> tci_val = in_table["Limiters"]["TCI_val"].value<Real>( );
  std::optional<bool> characteristic =
      in_table["Limiters"]["Characteristic1"].value<bool>( );

  if ( pn ) {
    ProblemName = pn.value( );
  } else {
    throw Error( " ! Error: problem not supplied in input deck." );
  }
  if ( bc ) {
    BC = bc.value( );
  } else {
    throw Error( " ! Error: boundary condition not supplied in input deck." );
  }
  if ( geom ) {
    Geometry = ( utilities::to_lower( geom.value( ) ) == "spherical" )
                   ? geometry::Spherical
                   : geometry::Planar;
  } else {
    Geometry = geometry::Planar;
  }
  if ( basis ) {
    Basis = ( utilities::to_lower( basis.value( ) ) == "legendre" )
                ? PolyBasis::Legendre
                : PolyBasis::Taylor;
  } else {
    Basis = PolyBasis::Legendre;
  }

  if ( x1 ) {
    xL = x1.value( );
  } else {
    throw Error( " ! Error: xL not supplied in input deck." );
  }
  if ( x2 ) {
    xR = x2.value( );
  } else {
    throw Error( " ! Error: xR not supplied in input deck." );
  }
  if ( tf ) {
    t_end = tf.value( );
  } else {
    throw Error( " ! Error: t_end not supplied in input deck." );
  }
  CFL = cfl.value_or( 0.5 );
  if ( nX ) {
    nElements = nX.value( );
  } else {
    throw Error( " ! Error: nX not supplied in innput deck." );
  }

  Restart = rest.value_or( false );
  do_rad  = rad.value_or( false );

  nGhost  = nG.value_or( 1 );
  pOrder  = pO.value_or( 1 );
  nNodes  = nN.value_or( 1 );
  tOrder  = tO.value_or( 1 );
  nStages = nS.value_or( 1 );

  alpha          = al.value_or( 1.0 );
  SL_Threshold   = slt.value_or( 1.0e-6 );
  TCI_Option     = tci_opt.value_or( false );
  TCI_Threshold  = tci_val.value_or( 0.1 );
  Characteristic = characteristic.value_or( false );
}
