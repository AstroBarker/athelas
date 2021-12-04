#ifndef TIMESTEPPER_H
#define TIMESTEPPER_H

typedef void myFuncType (DataStructure3D&, GridStructure&, ModalBasis&,
  DataStructure3D&, DataStructure3D&, DataStructure2D&, 
  DataStructure2D&, DataStructure2D&, std::vector<double>&, 
  std::vector<double>&, std::vector<double>, std::vector<double>,
  const std::string);

void InitializeTimestepper( const unsigned short int nStages, 
  DataStructure2D& a_jk, DataStructure2D& b_jk );

void UpdateFluid( myFuncType ComputeIncrement, double dt, 
  DataStructure3D& U, GridStructure& Grid, ModalBasis& Basis,
  DataStructure2D& a_jk, DataStructure2D& b_jk,
  std::vector<DataStructure3D>& U_s, std::vector<DataStructure3D>& dU_s, std::vector<GridStructure>& Grid_s,
  DataStructure3D& dU, DataStructure3D& SumVar, DataStructure3D& Flux_q, DataStructure2D& dFlux_num, 
  DataStructure2D& uCF_F_L, DataStructure2D& uCF_F_R, std::vector<std::vector<double>>& Flux_U, 
  std::vector<double>& Flux_P, std::vector<double> uCF_L, std::vector<double> uCF_R,
  const short unsigned int nStages, SlopeLimiter& S_Limiter,
  const std::string BC );

#endif