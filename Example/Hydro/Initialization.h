#ifndef INITIALIZATION_H
#define INITIALIZATION_H

void InitializeFields( DataStructure3D& uCF, DataStructure3D& uPF, 
  GridStructure& Grid, const unsigned int pOrder, const double GAMMA_IDEAL,
  const std::string ProblemName );
#endif
