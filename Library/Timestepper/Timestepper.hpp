#ifndef TIMESTEPPER_H
#define TIMESTEPPER_H

/**
 * File     :  Timestepper.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for SSPRK timestepping
 **/

#include "Abstractions.hpp"
#include "ProblemIn.hpp"

typedef void ( *UpdateFunc )( const Kokkos::View<Real ***>, GridStructure *,
                              ModalBasis *, Kokkos::View<Real ***>,
                              Kokkos::View<Real ***>, Kokkos::View<Real **>,
                              Kokkos::View<Real **>, Kokkos::View<Real **>,
                              Kokkos::View<Real *>, Kokkos::View<Real *>,
                              const std::string );

class TimeStepper
{

 public:
  // TODO: Is it possible to initialize Grid_s from Grid directly?
  TimeStepper( ProblemIn *pin, GridStructure *Grid );

  void InitializeTimestepper( );

  void UpdateFluid( UpdateFunc ComputeIncrement, const Real dt,
                    Kokkos::View<Real ***> U, GridStructure *Grid,
                    ModalBasis *Basis, SlopeLimiter *S_Limiter );

 private:
  const UInt mSize;
  const UInt nStages;
  const UInt tOrder;
  const std::string BC;

  // SSP coefficients
  Kokkos::View<Real **> a_jk;
  Kokkos::View<Real **> b_jk;

  // Hold stage data
  Kokkos::View<Real ****> U_s;
  Kokkos::View<Real ****> dU_s;
  Kokkos::View<Real ***> SumVar_U;
  std::vector<GridStructure> Grid_s;

  // StageData Holds cell left interface positions
  Kokkos::View<Real **> StageData;

  // Variables to pass to update step
  Kokkos::View<Real ***> Flux_q;

  Kokkos::View<Real **> dFlux_num;
  Kokkos::View<Real **> uCF_F_L;
  Kokkos::View<Real **> uCF_F_R;

  Kokkos::View<Real **> Flux_U;
  Kokkos::View<Real *> Flux_P;
};

#endif
