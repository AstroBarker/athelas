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
    double SlopeLimiterThreshold, unsigned int Beta_TVD_val, 
    unsigned int Beta_TVB_val, bool CharacteristicLimitingOption, 
    bool TCIOption, double TCI_Threshold_val );

  void ApplySlopeLimiter( DataStructure3D& U, GridStructure& Grid, 
    ModalBasis& Basis );

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
  }

private:

  unsigned int order;
  double SlopeLimiter_Threshold;
  unsigned int Beta_TVD, Beta_TVB;
  bool CharacteristicLimiting_Option;
  bool TCI_Option;
  double TCI_Threshold;

  double* R;
  double* R_inv;

  double* SlopeDifference;
  double* dU;

  DataStructure2D D;

};

#endif
