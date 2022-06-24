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
                       double* d2w, unsigned int iX, unsigned int nNodes );

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

  double R[9];
  double R_inv[9];

  double SlopeDifference[3];
  double dU[3];
  double d2U[3];
  double d2w[3];

  // --- Slope limiter quantities ---

  double U_c_L[3];
  double U_c_T[3];
  double U_c_R[3];
  double U_v_L[3];
  double U_v_R[3];

  double dU_c_L[3];
  double dU_c_T[3];
  double dU_c_R[3];
  double dU_v_L[3];
  double dU_v_R[3];

  // characteristic forms
  double w_c_L[3];
  double w_c_T[3];
  double w_c_R[3];
  double w_v_L[3];
  double w_v_R[3];

  double dw_c_L[3];
  double dw_c_T[3];
  double dw_c_R[3];
  double dw_v_L[3];
  double dw_v_R[3];

  // matrix mult scratch scape
  double Mult1[3];
  double Mult2[3];
  double Mult3[3];

  Kokkos::View<double**> D;
  Kokkos::View<int*> LimitedCell;
};

#endif
