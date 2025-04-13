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
auto State::Get_nCF( ) const -> int { return this->nCF; }
auto State::Get_nPF( ) const -> int { return this->nPF; }
auto State::Get_nAF( ) const -> int { return this->nAF; }
auto State::Get_nCR( ) const -> int { return this->nCR; }
auto State::Get_pOrder( ) const -> int { return this->pOrder; }

// view accessors
View3D<Real> State::Get_uCF( ) const { return this->uCF; }
View3D<Real> State::Get_uPF( ) const { return this->uPF; }
View3D<Real> State::Get_uAF( ) const { return this->uAF; }
View3D<Real> State::Get_uCR( ) const { return this->uCR; }
