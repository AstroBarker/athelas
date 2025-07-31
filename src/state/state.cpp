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
 */

#include "state.hpp"

State::State(const int nCF, const int nPF, const int nAF, const int nX_,
             const int nNodes_, const int pOrder)
    : nCF_(nCF), nPF_(nPF), nAF_(nAF), pOrder_(pOrder),
      uCF_("uCF", nCF_, nX_ + 2, pOrder_),
      uPF_("uPF", nPF_, nX_ + 2, nNodes_),
      uAF_("uAF", nAF_, nX_ + 2, nNodes_) {}

// num var accessors
auto State::get_n_cf() const noexcept -> int { return this->nCF_; }
auto State::get_n_pf() const noexcept -> int { return this->nPF_; }
auto State::get_n_af() const noexcept -> int { return this->nAF_; }
auto State::get_p_order() const noexcept -> int { return this->pOrder_; }

// view accessors
auto State::get_u_cf() const noexcept -> View3D<double> { return this->uCF_; }
auto State::get_u_pf() const noexcept -> View3D<double> { return this->uPF_; }
auto State::get_u_af() const noexcept -> View3D<double> { return this->uAF_; }
