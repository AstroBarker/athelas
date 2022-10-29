#ifndef TIMESTEPPER_H
#define TIMESTEPPER_H

/**
 * File     :  Timestepper.h
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for SSPRK timestepping
 **/

typedef void myFuncType( const Kokkos::View<double***>, const GridStructure&,
                         const ModalBasis&, Kokkos::View<double***>,
                         Kokkos::View<double***>, Kokkos::View<double**>,
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

  void UpdateFluid( myFuncType ComputeIncrement, const double dt,
                    Kokkos::View<double***> U, GridStructure& Grid,
                    const ModalBasis& Basis, SlopeLimiter& S_Limiter );

 private:
  const unsigned int mSize;
  const unsigned int nStages;
  const unsigned int tOrder;
  const std::string BC;

  // SSP coefficients
  Kokkos::View<double**> a_jk;
  Kokkos::View<double**> b_jk;

  Kokkos::View<double*> SumVar_X;

  // Hold stage data
  Kokkos::View<double****> U_s;
  Kokkos::View<double****> dU_s;
  Kokkos::View<double***> SumVar_U;
  std::vector<GridStructure> Grid_s;

  // StageData Holds cell left interface positions
  Kokkos::View<double**> StageData;

  // Variables to pass to update step
  Kokkos::View<double***> Flux_q;

  Kokkos::View<double**> dFlux_num;
  Kokkos::View<double**> uCF_F_L;
  Kokkos::View<double**> uCF_F_R;

  Kokkos::View<double**> Flux_U;
  Kokkos::View<double*> Flux_P;
};

#endif