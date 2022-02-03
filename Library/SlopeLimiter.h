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

#include <vector>
#include <iostream>

#include "Error.h"
#include "DataStructures.h"
#include "Grid.h"
#include "PolynomialBasis.h"


class SlopeLimiter
{

public:

  SlopeLimiter( GridStructure& Grid, unsigned int pOrder, 
    double SlopeLimiterThreshold, double alpha_val, 
    bool CharacteristicLimitingOption, 
    bool TCIOption, double TCI_Threshold_val );

  void ApplySlopeLimiter( DataStructure3D& U, GridStructure& Grid, 
    ModalBasis& Basis );

  void LimitQuadratic( DataStructure3D& U, ModalBasis& Basis, 
    double* d2w, unsigned int iX, unsigned int nNodes );

  void DetectTroubledCells( DataStructure3D& U, 
    GridStructure& Grid, ModalBasis& Basis );

  double CellAverage( DataStructure3D& U, GridStructure& Grid, ModalBasis& Basis,
    unsigned int iCF, unsigned int iX, int extrapolate );

  ~SlopeLimiter()
  {

  }

private:

  unsigned int order;
  double SlopeLimiter_Threshold;
  bool CharacteristicLimiting_Option;
  bool TCI_Option;
  double TCI_Threshold;

  double alpha;
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

  DataStructure2D D;

};

#endif
