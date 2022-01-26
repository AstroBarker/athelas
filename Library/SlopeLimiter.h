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
    delete [] R;
    delete [] R_inv;
    delete [] SlopeDifference;
    delete [] dU;
    delete [] d2U;
    delete [] d2w;
    delete [] U_c_L;
    delete [] U_c_T;
    delete [] U_c_R;
    delete [] U_v_L;
    delete [] U_v_R;
    delete [] dU_c_L;
    delete [] dU_c_T;
    delete [] dU_c_R;
    delete [] dU_v_L;
    delete [] dU_v_R;
    delete [] w_c_L;
    delete [] w_c_T;
    delete [] w_c_R;
    delete [] w_v_L;
    delete [] w_v_R;
    delete [] dw_c_L;
    delete [] dw_c_T;
    delete [] dw_c_R;
    delete [] dw_v_L;
    delete [] dw_v_R;
    delete [] Mult1;
    delete [] Mult2;
    delete [] Mult3;
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

  double* R;
  double* R_inv;

  double* SlopeDifference;
  double* dU;
  double* d2U;
  double* d2w;

  // --- Slope limiter quantities ---

  double* U_c_L;
  double* U_c_T;
  double* U_c_R;
  double* U_v_L;
  double* U_v_R;

  double* dU_c_L;
  double* dU_c_T;
  double* dU_c_R;
  double* dU_v_L;
  double* dU_v_R;

  // characteristic forms
  double* w_c_L;
  double* w_c_T;
  double* w_c_R;
  double* w_v_L;
  double* w_v_R;

  double* dw_c_L;
  double* dw_c_T;
  double* dw_c_R;
  double* dw_v_L;
  double* dw_v_R;

  // matrix mult scratch scape
  double* Mult1;
  double* Mult2;
  double* Mult3;

  DataStructure2D D;

};

#endif
