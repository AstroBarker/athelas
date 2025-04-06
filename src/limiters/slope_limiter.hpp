#ifndef SLOPE_LIMITER_HPP_
#define SLOPE_LIMITER_HPP_

/**
 * Specific slope limiter classes here
 **/

#include <variant>

#include "abstractions.hpp"
#include "error.hpp"
#include "slope_limiter_base.hpp"

class WENO : public SlopeLimiterBase<WENO> {
 public:
  WENO( ) = default;
  WENO( const GridStructure *grid, const ProblemIn *pin, const int nvars_ )
      : do_limiter( pin->do_limiter ), order( pin->pOrder ), nvars( nvars_ ),
        gamma_i( pin->gamma_i ), gamma_l( pin->gamma_l ),
        gamma_r( pin->gamma_r ), weno_r( pin->weno_r ),
        characteristic( pin->Characteristic ), tci_opt( pin->TCI_Option ),
        tci_val( pin->TCI_Threshold ),
        modified_polynomial( "modified_polynomial", grid->Get_nElements( ) + 2,
                             3, pin->pOrder ),
        R( "R Matrix", 3, 3, grid->Get_nElements( ) + 2 ),
        R_inv( "invR Matrix", 3, 3, grid->Get_nElements( ) + 2 ),
        U_c_T( "U_c_T", 3, grid->Get_nElements( ) + 2 ),
        w_c_T( "w_c_T", 3, grid->Get_nElements( ) + 2 ),
        Mult( "Mult", 3, grid->Get_nElements( ) + 2 ),
        D( "TCI", 3, grid->Get_nElements( ) + 2 ),
        LimitedCell( "LimitedCell", grid->Get_nElements( ) + 2 ) {}

  void ApplySlopeLimiter( View3D<Real> U, const GridStructure *grid,
                          const ModalBasis *basis );
  int Get_Limited( const int iX ) const;

 private:
  bool do_limiter;
  int order;
  int nvars;
  Real gamma_i;
  Real gamma_l;
  Real gamma_r;
  Real weno_r;
  bool characteristic;
  bool tci_opt;
  Real tci_val;

  View3D<Real> modified_polynomial;

  View3D<Real> R;
  View3D<Real> R_inv;

  // --- Slope limiter quantities ---

  View2D<Real> U_c_T;

  // characteristic forms
  View2D<Real> w_c_T;

  // matrix mult scratch scape
  View2D<Real> Mult;

  View2D<Real> D;
  View1D<int> LimitedCell;
};

class TVDMinmod : public SlopeLimiterBase<TVDMinmod> {
 public:
  TVDMinmod( ) = default;
  TVDMinmod( const GridStructure *grid, const ProblemIn *pin, const int nvars_ )
      : do_limiter( pin->do_limiter ), order( pin->pOrder ), nvars( nvars_ ),
        b_tvd( pin->b_tvd ), m_tvb( pin->m_tvb ),
        characteristic( pin->Characteristic ), tci_opt( pin->TCI_Option ),
        tci_val( pin->TCI_Threshold ),
        R( "R Matrix", 3, 3, grid->Get_nElements( ) + 2 * grid->Get_Guard( ) ),
        R_inv( "invR Matrix", 3, 3,
               grid->Get_nElements( ) + 2 * grid->Get_Guard( ) ),
        U_c_T( "U_c_T", 3, grid->Get_nElements( ) + 2 ),
        w_c_T( "w_c_T", 3, grid->Get_nElements( ) + 2 ),
        Mult( "Mult", 3, grid->Get_nElements( ) + 2 ),
        D( "TCI", 3, grid->Get_nElements( ) + 2 * grid->Get_Guard( ) ),
        LimitedCell( "LimitedCell",
                     grid->Get_nElements( ) + 2 * grid->Get_Guard( ) ) {}
  void ApplySlopeLimiter( View3D<Real> U, const GridStructure *grid,
                          const ModalBasis *basis );
  int Get_Limited( const int iX ) const;

 private:
  bool do_limiter;
  int order;
  int nvars;
  Real b_tvd;
  Real m_tvb;
  bool characteristic;
  bool tci_opt;
  Real tci_val;

  View3D<Real> R;
  View3D<Real> R_inv;

  // --- Slope limiter quantities ---

  View2D<Real> U_c_T;

  // characteristic forms
  View2D<Real> w_c_T;

  // matrix mult scratch scape
  View2D<Real> Mult;

  View2D<Real> D;
  View1D<int> LimitedCell;
};

// TODO: adjust when we support more than one SLOPE_LIMITER
// using SlopeLimiter = WENO;
using SlopeLimiter = std::variant<WENO, TVDMinmod>;

// std::visit functions
KOKKOS_INLINE_FUNCTION void ApplySlopeLimiter( SlopeLimiter *limiter,
                                               View3D<Real> U,
                                               const GridStructure *grid,
                                               const ModalBasis *basis ) {
  std::visit(
      [&U, &grid, &basis]( auto &limiter ) {
        limiter.ApplySlopeLimiter( U, grid, basis );
      },
      *limiter );
}
KOKKOS_INLINE_FUNCTION int Get_Limited( SlopeLimiter *limiter, const int iX ) {
  return std::visit(
      [&iX]( auto &limiter ) { return limiter.Get_Limited( iX ); }, *limiter );
}

#endif // SLOPE_LIMITER_HPP_
