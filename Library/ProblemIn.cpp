/**
 * File     :  ProblemIn.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the problem initialization
 *
 **/

#include "ProblemIn.hpp"
#include "Constants.h"
#include "Error.h"
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

  nN = ini.GetValue("Grid", "nNodes");
  nX = ini.GetValue("Grid", "nX");
  nG = ini.GetValue("Grid", "nG");
  pO = ini.GetValue("Grid", "pOrder");
  tO = ini.GetValue("Grid", "tOrder");
  nS = ini.GetValue("Grid", "nStages");

  ProblemName = pn;
  BC          = bc;
  Restart     = (strcmp(rest, "true") == 0) ? true : false;
  Geometry    = (strcmp(geom, "true") == 0) ? true : false;
  xL          = std::atof( x1 );
  xR          = std::atof( x2 );
  t_end       = std::atof( tf );
  CFL         = std::atof( cfl );

  nNodes    = std::atoi(nN);
  nElements = std::atoi( nX );
  nGhost    = std::atoi( nG );
  pOrder    = std::atoi( pO );
  tOrder    = std::atoi( tO );
  nStages   = std::atoi( nS );
}
