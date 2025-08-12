#include "state.hpp"

State::State(const int nCF, const int nPF, const int nAF, const int nX_,
             const int nNodes_, const int pOrder,
             const bool composition_enabled, const int ncomps)
    : nCF_(nCF), nPF_(nPF), nAF_(nAF), pOrder_(pOrder),
      uCF_("uCF", nCF_, nX_ + 2, pOrder_), uPF_("uPF", nPF_, nX_ + 2, nNodes_),
      uAF_("uAF", nAF_, nX_ + 2, nNodes_),
      composition_enabled_(composition_enabled) {
  if (composition_enabled) {
    if (ncomps <= 0) {
      THROW_ATHELAS_ERROR("Composition enabled but ncomps <= 0!");
    }
    uComp_ = View3D<double>("composition", ncomps, nX_, nNodes_);
  }
}

// num var accessors
auto State::n_cf() const noexcept -> int { return nCF_; }
auto State::n_pf() const noexcept -> int { return nPF_; }
auto State::n_af() const noexcept -> int { return nAF_; }
auto State::p_order() const noexcept -> int { return pOrder_; }

// view accessors
auto State::u_cf() const noexcept -> View3D<double> { return uCF_; }
auto State::u_pf() const noexcept -> View3D<double> { return uPF_; }
auto State::u_af() const noexcept -> View3D<double> { return uAF_; }
