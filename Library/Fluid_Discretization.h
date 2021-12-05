#ifndef FLUID_DISCRETIZATION_H
#define FLUID_DISCRETIZATION_H

void ComputeIncrement_Fluid_Divergence( DataStructure3D& U, GridStructure& Grid, 
  ModalBasis& Basis, DataStructure3D& dU, DataStructure3D& Flux_q, 
  DataStructure2D& dFlux_num, DataStructure2D& uCF_F_L, 
  DataStructure2D& uCF_F_R, std::vector<double>& Flux_U, 
  std::vector<double>& Flux_P, std::vector<double> uCF_L, std::vector<double> uCF_R );


void Compute_Increment_Explicit( DataStructure3D& U, GridStructure& Grid, 
  ModalBasis& Basis, DataStructure3D& dU, DataStructure3D& Flux_q, 
  DataStructure2D& dFlux_num, DataStructure2D& uCF_F_L, 
  DataStructure2D& uCF_F_R, std::vector<double>& Flux_U, 
  std::vector<double>& Flux_P, std::vector<double> uCF_L, 
  std::vector<double> uCF_R, const std::string BC );

#endif