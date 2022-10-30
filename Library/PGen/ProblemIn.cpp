/**
 * File     :  ProblemIn.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the problem initialization
 *
 **/

#include "ProblemIn.hpp"
#include "Error.hpp"
#include "SimpleIni.h" 

ProblemIn::ProblemIn( std::string fn ) 
{
  // Load ini
  CSimpleIniA ini;
  SI_Error rc = ini.LoadFile( fn.c_str() );

  // Check if everything is happy
  if (rc < 0) { throw Error("Problem reading input deck\n"); }
  assert( rc == SI_OK );

  // Grab as char*, cast to respective types
  const char* nN;
  const char* nX;
  const char* nG;
  const char* pO;
  const char* tO;
  const char* nS;
  const char* rest;
  const char* geom;
  const char* pn;
  const char* tf;
  const char* x1;
  const char* x2;
  const char* bc;
  const char* cfl;

  pn = ini.GetValue("Problem", "problem");
  rest = ini.GetValue("Problem", "restart");
  geom = ini.GetValue("Problem", "geometry");
  tf = ini.GetValue("Problem", "t_end");
  x1 = ini.GetValue("Problem", "xL");
  x2 = ini.GetValue("Problem", "xR");
  bc = ini.GetValue("Problem", "BC");
  cfl = ini.GetValue("Problem", "CFL");

  nN = ini.GetValue("Fluid", "nNodes");
  nX = ini.GetValue("Fluid", "nX");
  nG = ini.GetValue("Fluid", "nG");
  pO = ini.GetValue("Fluid", "pOrder");

  tO = ini.GetValue("Time", "tOrder");
  nS = ini.GetValue("Time", "nStages");

  ProblemName = pn;
  BC          = bc;
  Restart     = (strcmp(rest, "true") == 0) ? true : false;
  Geometry    = (strcmp(geom, "spherical") == 0) ? geometry::Spherical : geometry::Planar;
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
}