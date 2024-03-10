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

  Real ModifyPolynomial( const View3D U, const ModalBasis *Basis,
                         const Real Ubar_i, const int iX, const int iCQ,
                         const int iN );
  void ModifyPolynomial( const View3D U, const int iX, const int iCQ );
  Real SmoothnessIndicator( const View3D U, const GridStructure *Grid,
                            const int iX, const int i, const int iCQ );
  Real NonLinearWeight( const Real gamma, const Real beta, const Real tau,
                        const Real eps );
  Real Tau( const Real beta_l, const Real beta_i, const Real beta_r );

  void ApplySlopeLimiter( View3D U, GridStructure *Grid,
                          const ModalBasis *Basis );

  void LimitQuadratic( View3D U, const ModalBasis *Basis,
                       Kokkos::View<Real[3]> d2w, const int iX,
                       const int nNodes );

  void DetectTroubledCells( View3D U, GridStructure *Grid,
                            const ModalBasis *Basis );

  Real CellAverage( View3D U, GridStructure *Grid, const ModalBasis *Basis,
                    const int iCF, const int iX, const int extrapolate );

  int Get_Limited( int iX ) const;

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

  View2D modified_polynomial;

  View3D R;
  View3D R_inv;

  // --- Slope limiter quantities ---

  View1D U_c_T;

  // characteristic forms
  View1D w_c_T;

  // matrix mult scratch scape
  View1D Mult;

  Kokkos::View<Real **> D;
  Kokkos::View<int *> LimitedCell;
};

#endif // _SLOPELIMITER_HPP_
