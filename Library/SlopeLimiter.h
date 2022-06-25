#ifndef SLOPELIMITER_H
#define SLOPELIMITER_H

/**
 * File     :  SlopeLimiter.h
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding slope limiter data and routines.
 * Contains : SlopeLimiter
 **/

#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "Error.h"
#include "Grid.h"
#include "PolynomialBasis.h"

class SlopeLimiter
{

 public:
  SlopeLimiter( GridStructure& Grid, unsigned int pOrder,
                double SlopeLimiterThreshold, double alpha_val,
                bool CharacteristicLimitingOption, bool TCIOption,
                double TCI_Threshold_val );

  void ApplySlopeLimiter( Kokkos::View<double***> U, GridStructure& Grid,
                          ModalBasis& Basis );

  void LimitQuadratic( Kokkos::View<double***> U, ModalBasis& Basis,
                       Kokkos::View<double[3]> d2w, unsigned int iX,
                       unsigned int nNodes );

  void DetectTroubledCells( Kokkos::View<double***> U, GridStructure& Grid,
                            ModalBasis& Basis );

  double CellAverage( Kokkos::View<double***> U, GridStructure& Grid,
                      ModalBasis& Basis, unsigned int iCF, unsigned int iX,
                      int extrapolate );

  int Get_Limited( unsigned int iX ) const;

  ~SlopeLimiter( ) {}

 private:
  unsigned int order;
  double SlopeLimiter_Threshold;
  double alpha;
  bool CharacteristicLimiting_Option;
  bool TCI_Option;
  double TCI_Threshold;

  double Phi1;
  double Phi2;

  Kokkos::View<double[3][3]> R;
  Kokkos::View<double[3][3]> R_inv;

  Kokkos::View<double[3]> SlopeDifference;
  Kokkos::View<double[3]> dU;
  Kokkos::View<double[3]> d2U;
  Kokkos::View<double[3]> d2w;

  // --- Slope limiter quantities ---

  Kokkos::View<double[3]> U_c_L;
  Kokkos::View<double[3]> U_c_T;
  Kokkos::View<double[3]> U_c_R;
  Kokkos::View<double[3]> U_v_L;
  Kokkos::View<double[3]> U_v_R;

  Kokkos::View<double[3]> dU_c_L;
  Kokkos::View<double[3]> dU_c_T;
  Kokkos::View<double[3]> dU_c_R;
  Kokkos::View<double[3]> dU_v_L;
  Kokkos::View<double[3]> dU_v_R;

  // characteristic forms
  Kokkos::View<double[3]> w_c_L;
  Kokkos::View<double[3]> w_c_T;
  Kokkos::View<double[3]> w_c_R;
  Kokkos::View<double[3]> w_v_L;
  Kokkos::View<double[3]> w_v_R;

  Kokkos::View<double[3]> dw_c_L;
  Kokkos::View<double[3]> dw_c_T;
  Kokkos::View<double[3]> dw_c_R;
  Kokkos::View<double[3]> dw_v_L;
  Kokkos::View<double[3]> dw_v_R;

  // matrix mult scratch scape
  Kokkos::View<double[3]> Mult1;
  Kokkos::View<double[3]> Mult2;
  Kokkos::View<double[3]> Mult3;

  Kokkos::View<double**> D;
  Kokkos::View<int*> LimitedCell;
};

#endif
