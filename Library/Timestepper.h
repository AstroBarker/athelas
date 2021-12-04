#ifndef TIMESTEPPER_H
#define TIMESTEPPER_H

/**
 * File     :  Timestepper.h
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for SSPRK timestepping
**/ 

typedef void myFuncType (DataStructure3D&, GridStructure&, ModalBasis&,
  DataStructure3D&, DataStructure3D&, DataStructure2D&, 
  DataStructure2D&, DataStructure2D&, std::vector<double>&, 
  std::vector<double>&, std::vector<double>, std::vector<double>,
  const std::string);

class TimeStepper
{

public:

  TimeStepper( unsigned int nS, unsigned int tO, unsigned int pO,
    GridStructure& Grid, std::string BCond );

  void InitializeTimestepper( );

  void UpdateFluid( myFuncType ComputeIncrement, double dt, 
    DataStructure3D& U, GridStructure& Grid, ModalBasis& Basis,
    SlopeLimiter& S_Limiter );

private: 

  const unsigned int mSize;
  const unsigned int nStages;
  const unsigned int tOrder;
  const std::string BC;

  // SSP coefficients
  DataStructure2D a_jk;
  DataStructure2D b_jk;

  // Summations
  DataStructure3D SumVar_U;
  std::vector<double> SumVar_X;

  // Hold stage data
  std::vector<DataStructure3D> U_s;
  std::vector<DataStructure3D> dU_s;
  std::vector<GridStructure> Grid_s;

  // Variables to pass to update step
  DataStructure3D Flux_q;

  DataStructure2D dFlux_num;
  DataStructure2D uCF_F_L;
  DataStructure2D uCF_F_R;

  std::vector<std::vector<double>> Flux_U;
  std::vector<double> Flux_P;

  std::vector<double> uCF_L;
  std::vector<double> uCF_R;

};

#endif