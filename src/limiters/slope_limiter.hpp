#ifndef SLOPE_LIMITER_HPP_
#define SLOPE_LIMITER_HPP_
/**
 * @file slope_limiter.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Specific slope limiter classes that implement the
 *        SlopeLimiterBase interface
 *
 * @details Defines specific slope limiter implementations that
 *          inherit from the SlopeLimiterBase template class.
 *
 *          We implement the following limiters:
 *          - WENO: Weighted Essentially Non-Oscillatory limiter
 *          - TVDMinmod: Total Variation Diminishing Minmod limiter
 *
 *          Both limiters support:
 *          - Characteristic decomposition
 *          - Troubled Cell Indicator (TCI)
 */

#include <variant>

#include "abstractions.hpp"
#include "error.hpp"
#include "slope_limiter_base.hpp"

class WENO : public SlopeLimiterBase<WENO> {
 public:
  WENO( ) = default;
  WENO( const GridStructure* grid, const ProblemIn* pin, const int nvars_ )
      : do_limiter( pin->do_limiter ), order( pin->pOrder ), nvars( nvars_ ),
        gamma_i( pin->gamma_i ), gamma_l( pin->gamma_l ),
        gamma_r( pin->gamma_r ), weno_r( pin->weno_r ),
        characteristic( pin->Characteristic ), tci_opt( pin->TCI_Option ),
        tci_val( pin->TCI_Threshold ),
        modified_polynomial( "modified_polynomial", grid->get_n_elements( ) + 2,
                             3, pin->pOrder ),
        R( "R Matrix", 3, 3, grid->get_n_elements( ) + 2 ),
        R_inv( "invR Matrix", 3, 3, grid->get_n_elements( ) + 2 ),
        U_c_T( "U_c_T", 3, grid->get_n_elements( ) + 2 ),
        w_c_T( "w_c_T", 3, grid->get_n_elements( ) + 2 ),
        Mult( "Mult", 3, grid->get_n_elements( ) + 2 ),
        D( "TCI", 3, grid->get_n_elements( ) + 2 ),
        LimitedCell( "LimitedCell", grid->get_n_elements( ) + 2 ) {}

  void apply_slope_limiter( View3D<Real> U, const GridStructure* grid,
                            const ModalBasis* basis );
  [[nodiscard]] auto get_limited( int iX ) const -> int;

 private:
  bool do_limiter{ };
  int order{ };
  int nvars{ };
  Real gamma_i{ };
  Real gamma_l{ };
  Real gamma_r{ };
  Real weno_r{ };
  bool characteristic{ };
  bool tci_opt{ };
  Real tci_val{ };

  View3D<Real> modified_polynomial{ };

  View3D<Real> R{ };
  View3D<Real> R_inv{ };

  // --- Slope limiter quantities ---

  View2D<Real> U_c_T{ };

  // characteristic forms
  View2D<Real> w_c_T{ };

  // matrix mult scratch scape
  View2D<Real> Mult{ };

  View2D<Real> D{ };
  View1D<int> LimitedCell{ };
};

class TVDMinmod : public SlopeLimiterBase<TVDMinmod> {
 public:
  TVDMinmod( ) = default;
  TVDMinmod( const GridStructure* grid, const ProblemIn* pin, const int nvars_ )
      : do_limiter( pin->do_limiter ), order( pin->pOrder ), nvars( nvars_ ),
        b_tvd( pin->b_tvd ), m_tvb( pin->m_tvb ),
        characteristic( pin->Characteristic ), tci_opt( pin->TCI_Option ),
        tci_val( pin->TCI_Threshold ),
        R( "R Matrix", 3, 3, grid->get_n_elements( ) + 2 * grid->get_guard( ) ),
        R_inv( "invR Matrix", 3, 3,
               grid->get_n_elements( ) + 2 * grid->get_guard( ) ),
        U_c_T( "U_c_T", 3, grid->get_n_elements( ) + 2 ),
        w_c_T( "w_c_T", 3, grid->get_n_elements( ) + 2 ),
        Mult( "Mult", 3, grid->get_n_elements( ) + 2 ),
        D( "TCI", 3, grid->get_n_elements( ) + 2 * grid->get_guard( ) ),
        LimitedCell( "LimitedCell",
                     grid->get_n_elements( ) + 2 * grid->get_guard( ) ) {}
  void apply_slope_limiter( View3D<Real> U, const GridStructure* grid,
                            const ModalBasis* basis );
  [[nodiscard]] auto get_limited( int iX ) const -> int;

 private:
  bool do_limiter{ };
  int order{ };
  int nvars{ };
  Real b_tvd{ };
  Real m_tvb{ };
  bool characteristic{ };
  bool tci_opt{ };
  Real tci_val{ };

  View3D<Real> R{ };
  View3D<Real> R_inv{ };

  // --- Slope limiter quantities ---

  View2D<Real> U_c_T{ };

  // characteristic forms
  View2D<Real> w_c_T{ };

  // matrix mult scratch scape
  View2D<Real> Mult{ };

  View2D<Real> D{ };
  View1D<int> LimitedCell{ };
};

// TODO(astrobarker): adjust when we support more than one SLOPE_LIMITER
// using SlopeLimiter = WENO;
using SlopeLimiter = std::variant<WENO, TVDMinmod>;

// std::visit functions
KOKKOS_INLINE_FUNCTION void apply_slope_limiter( SlopeLimiter* limiter,
                                                 View3D<Real> U,
                                                 const GridStructure* grid,
                                                 const ModalBasis* basis ) {
  std::visit(
      [&U, &grid, &basis]( auto& limiter ) {
        limiter.apply_slope_limiter( U, grid, basis );
      },
      *limiter );
}
KOKKOS_INLINE_FUNCTION int get_limited( SlopeLimiter* limiter, const int iX ) {
  return std::visit(
      [&iX]( auto& limiter ) { return limiter.get_limited( iX ); }, *limiter );
}

#endif // SLOPE_LIMITER_HPP_
