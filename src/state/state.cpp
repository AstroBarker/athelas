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

State::State( const int nCF, const int nCR, const int nPF, const int nAF,
              const int nX_, const int nG_, const int nNodes_,
              const int pOrder )
    : nCF_( nCF ), nCR_( nCR ), nPF_( nPF ), nAF_( nAF ), pOrder_( pOrder ),
      uCF_( "uCF", nCF_, nX_ + 2 * nG_, pOrder_ ),
      uPF_( "uPF", nPF_, nX_ + 2 * nG_, nNodes_ ),
      uAF_( "uAF", nAF_, nX_ + 2 * nG_, nNodes_ ),
      uCR_( "uCR", nCR_, nX_ + 2 * nG_, pOrder_ ) {}

// num var accessors
auto State::get_n_cf( ) const noexcept -> int { return this->nCF_; }
auto State::get_n_pf( ) const noexcept -> int { return this->nPF_; }
auto State::get_n_af( ) const noexcept -> int { return this->nAF_; }
auto State::get_n_cr( ) const noexcept -> int { return this->nCR_; }
auto State::get_p_order( ) const noexcept -> int { return this->pOrder_; }

// view accessors
View3D<double> State::get_u_cf( ) const noexcept { return this->uCF_; }
View3D<double> State::get_u_pf( ) const noexcept { return this->uPF_; }
View3D<double> State::get_u_af( ) const noexcept { return this->uAF_; }
View3D<double> State::get_u_cr( ) const noexcept { return this->uCR_; }
