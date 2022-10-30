#ifndef PROBLEMIN_H
#define PROBLEMIN_H

/**
 * File     :  ProblemIn.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the problem intialization 
 *
 **/

#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Geometry.hpp"
#include "Error.hpp"
#include "SimpleIni.h"

class ProblemIn
{
 public:
  ProblemIn( std::string fn );

  std::string ProblemName;
  std::string BC;

  UInt nElements;
  UInt nNodes;
  UInt nGhost;

  UInt pOrder;
  UInt tOrder;
  UInt nStages;

  Real xL;
  Real xR;
  Real CFL;

  Real t_end;

  geometry::Geometry Geometry;
  bool Restart;

  Real alpha;
  Real SL_Threshold;
  bool TCI_Option;
  Real TCI_Threshold;
  bool Characteristic;
};

#endif
