#ifndef SLOPELIMITER_HPP_
#define SLOPELIMITER_HPP_

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

  void ModifyPolynomial( const View3D<Real> U, const int iX, const int iCQ );
  Real SmoothnessIndicator( const View3D<Real> U, const GridStructure *Grid,
                            const int iX, const int i, const int iCQ ) const;
  Real NonLinearWeight( const Real gamma, const Real beta, const Real tau,
                        const Real eps ) const;
  Real Tau( const Real beta_l, const Real beta_i, const Real beta_r ) const;

  void ApplySlopeLimiter( View3D<Real> U, GridStructure *Grid,
                          const ModalBasis *Basis );

  void DetectTroubledCells( View3D<Real> U, GridStructure *Grid,
                            const ModalBasis *Basis );

  Real CellAverage( View3D<Real> U, GridStructure *Grid,
                    const ModalBasis *Basis, const int iCF, const int iX,
                    const int extrapolate ) const;

  int Get_Limited( const int iX ) const;

  ~SlopeLimiter( ) {}

 private:
  int order;
  bool CharacteristicLimiting_Option;
  bool TCI_Option;
  Real TCI_Threshold;

  // TODO: from input deck
  Real gamma_l;
  Real gamma_i;
  Real gamma_r;
  Real weno_r; // nonlinear weight power

  View2D<Real> modified_polynomial;

  View3D<Real> R;
  View3D<Real> R_inv;

  // --- Slope limiter quantities ---

  View1D<Real> U_c_T;

  // characteristic forms
  View1D<Real> w_c_T;

  // matrix mult scratch scape
  View1D<Real> Mult;

  View2D<Real> D;
  View1D<int> LimitedCell;
};

#endif // SLOPELIMITER_HPP_
