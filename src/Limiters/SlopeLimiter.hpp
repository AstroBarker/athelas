#ifndef _SLOPELIMITER_HPP_
#define _SLOPELIMITER_HPP_

/**
 * File     :  SlopeLimiter.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding slope limiter data and routines.
 * Contains : SlopeLimiter
 **/

#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Error.hpp"
#include "Grid.hpp"
#include "PolynomialBasis.hpp"
#include "ProblemIn.hpp"

class SlopeLimiter {

 public:
  SlopeLimiter( GridStructure *Grid, ProblemIn *pin );

  void ApplySlopeLimiter( View3D U, GridStructure *Grid,
                          const ModalBasis *Basis );

  void LimitQuadratic( View3D U, const ModalBasis *Basis,
                       Kokkos::View<Real[3]> d2w, const UInt iX,
                       const UInt nNodes );

  void DetectTroubledCells( View3D U, GridStructure *Grid,
                            const ModalBasis *Basis );

  Real CellAverage( View3D U, GridStructure *Grid, const ModalBasis *Basis,
                    const UInt iCF, const UInt iX, const int extrapolate );

  int Get_Limited( UInt iX ) const;

  ~SlopeLimiter( ) {}

 private:
  UInt order;
  Real SlopeLimiter_Threshold;
  Real alpha;
  bool CharacteristicLimiting_Option;
  bool TCI_Option;
  Real TCI_Threshold;

  Real Phi1;
  Real Phi2;

  Kokkos::View<Real[3][3]> R;
  Kokkos::View<Real[3][3]> R_inv;

  Kokkos::View<Real[3]> SlopeDifference;
  Kokkos::View<Real[3]> dU;
  Kokkos::View<Real[3]> d2U;
  Kokkos::View<Real[3]> d2w;

  // --- Slope limiter quantities ---

  Kokkos::View<Real[3]> U_c_L;
  Kokkos::View<Real[3]> U_c_T;
  Kokkos::View<Real[3]> U_c_R;
  Kokkos::View<Real[3]> U_v_L;
  Kokkos::View<Real[3]> U_v_R;

  Kokkos::View<Real[3]> dU_c_L;
  Kokkos::View<Real[3]> dU_c_T;
  Kokkos::View<Real[3]> dU_c_R;
  Kokkos::View<Real[3]> dU_v_L;
  Kokkos::View<Real[3]> dU_v_R;

  // characteristic forms
  Kokkos::View<Real[3]> w_c_L;
  Kokkos::View<Real[3]> w_c_T;
  Kokkos::View<Real[3]> w_c_R;
  Kokkos::View<Real[3]> w_v_L;
  Kokkos::View<Real[3]> w_v_R;

  Kokkos::View<Real[3]> dw_c_L;
  Kokkos::View<Real[3]> dw_c_T;
  Kokkos::View<Real[3]> dw_c_R;
  Kokkos::View<Real[3]> dw_v_L;
  Kokkos::View<Real[3]> dw_v_R;

  // matrix mult scratch scape
  Kokkos::View<Real[3]> Mult1;
  Kokkos::View<Real[3]> Mult2;
  Kokkos::View<Real[3]> Mult3;

  Kokkos::View<Real **> D;
  Kokkos::View<int *> LimitedCell;
};

#endif // _SLOPELIMITER_HPP_
