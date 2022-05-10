#ifndef TIMESTEPPER_H
#define TIMESTEPPER_H

/**
 * File     :  Timestepper.h
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for SSPRK timestepping
 **/


typedef void myFuncType( Kokkos::View<double***>, GridStructure&, ModalBasis&,
                         Kokkos::View<double***>, Kokkos::View<double***>, Kokkos::View<double**>,
                         Kokkos::View<double**>, Kokkos::View<double**>,
                         Kokkos::View<double*>, Kokkos::View<double*>,
                         const std::string );

class TimeStepper
{

 public:
  // TODO: Is it possible to initialize Grid_s from Grid directly?
  TimeStepper( unsigned int nS, unsigned int tO, unsigned int pO,
               GridStructure& Grid, bool Geometry, std::string BCond );

  void InitializeTimestepper( );

  void UpdateFluid( myFuncType ComputeIncrement, double dt, Kokkos::View<double***> U,
                    GridStructure& Grid, ModalBasis& Basis,
                    SlopeLimiter& S_Limiter );

 private:
  const unsigned int mSize;
  const unsigned int nStages;
  const unsigned int tOrder;
  const std::string BC;

  // SSP coefficients
  Kokkos::View<double**> a_jk;
  Kokkos::View<double**> b_jk;

  // Summations
  Kokkos::View<double***> SumVar_U;
  Kokkos::View<double*> SumVar_X;

  // Hold stage data
  Kokkos::View<double****> U_s;
  Kokkos::View<double****> dU_s;
  std::vector<GridStructure> Grid_s;
  Kokkos::View<double**> StageData;
  // StageData Holds cell left interface positions

  // Variables to pass to update step
  Kokkos::View<double***> Flux_q;

  Kokkos::View<double**> dFlux_num;
  Kokkos::View<double**> uCF_F_L;
  Kokkos::View<double**> uCF_F_R;

  Kokkos::View<double**> Flux_U;
  Kokkos::View<double*> Flux_P;

};

#endif