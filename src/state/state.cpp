/**
 * @file state.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Class for holding state data
 *
 * @details Contains:
 *          - uCF
 *          - uPF
 *          - uAF
 *          - uCR
 */

#include "state.hpp"
#include "constants.hpp"

State::State( const int nCF_, const int nCR_, const int nPF_, const int nAF_,
              const int nX_, const int nG_, const int nNodes_,
              const int pOrder_ )
    : nCF( nCF_ ), nCR( nCR_ ), nPF( nPF_ ), nAF( nAF_ ), pOrder( pOrder_ ),
      uCF( "uCF", nCF_, nX_ + 2 * nG_, pOrder_ ),
      uPF( "uPF", nPF_, nX_ + 2 * nG_, nNodes_ ),
      uAF( "uAF", nAF_, nX_ + 2 * nG_, nNodes_ ),
      uCR( "uCR", nCR_, nX_ + 2 * nG_, pOrder_ ) {}

// num var accessors
auto State::get_n_cf( ) const -> int { return this->nCF; }
auto State::get_n_pf( ) const -> int { return this->nPF; }
auto State::get_n_af( ) const -> int { return this->nAF; }
auto State::get_n_cr( ) const -> int { return this->nCR; }
auto State::get_p_order( ) const -> int { return this->pOrder; }

// view accessors
View3D<Real> State::get_u_cf( ) const { return this->uCF; }
View3D<Real> State::get_u_pf( ) const { return this->uPF; }
View3D<Real> State::get_u_af( ) const { return this->uAF; }
View3D<Real> State::get_u_cr( ) const { return this->uCR; }
