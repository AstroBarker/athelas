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


// This will be e.g., conserved variables
class SlopeLimiter
{

public:

  SlopeLimiter( GridStructure& Grid, unsigned int numNodes, 
    double SlopeLimiterThreshold, unsigned int Beta_TVD_val, 
    unsigned int Beta_TVB_val, bool CharacteristicLimitingOption, 
    bool TCIOption, double TCI_Threshold_val );

  void ApplySlopeLimiter( DataStructure3D& U, GridStructure& Grid, 
    DataStructure3D& D );

  ~SlopeLimiter()
  {
    delete [] R;
    delete [] R_inv;
    delete [] SlopeDifference;
    delete [] dU;
  }

private:

  unsigned int nNodes;
  double SlopeLimiter_Threshold;
  unsigned int Beta_TVD, Beta_TVB;
  bool CharacteristicLimiting_Option;
  bool TCI_Option;
  double TCI_Threshold;

  double* R;
  double* R_inv;

  double* SlopeDifference;
  double* dU;

};

#endif
