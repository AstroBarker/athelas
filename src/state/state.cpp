#include "state/state.hpp"
#include "utils/error.hpp"

State::State(const int nvar, const int nPF, const int nAF, const int nX_,
             const int nNodes_, const int pOrder, const int nstages,
             const bool composition_enabled, const bool ionization_enabled)
    : nvar_(nvar), nPF_(nPF), nAF_(nAF), pOrder_(pOrder),
      uCF_("uCF", nX_ + 2, pOrder_, nvar_),
      uCF_s_("uCF_s", nstages, nX_ + 2, pOrder_, nvar_),
      uPF_("uPF", nX_ + 2, nNodes_ + 2, nPF_),
      uAF_("uAF", nX_ + 2, nNodes_ + 2, nAF_),
      composition_enabled_(composition_enabled),
      ionization_enabled_(ionization_enabled) {}

void State::setup_composition(std::shared_ptr<CompositionData> comps) {
  if (!composition_enabled_) {
    THROW_ATHELAS_ERROR(
        "Trying to set composition but composition is not enabled!");
  }
  comps_ = std::move(comps);
}

void State::setup_ionization(std::shared_ptr<IonizationState> ion) {
  if (!ionization_enabled_) {
    THROW_ATHELAS_ERROR(
        "Trying to set ionization but ionization is not enabled!");
  }
  ionization_state_ = std::move(ion);
}

[[nodiscard]] auto State::comps() const -> CompositionData * {
  if (!composition_enabled_) {
    THROW_ATHELAS_ERROR("Composition not enabled!");
  }
  return comps_.get();
}

[[nodiscard]] auto State::ionization_state() const -> IonizationState * {
  if (!ionization_enabled_) {
    THROW_ATHELAS_ERROR("Ionization not enabled!");
  }
  return ionization_state_.get();
}

[[nodiscard]] auto State::composition_enabled() const noexcept -> bool {
  return composition_enabled_;
}

[[nodiscard]] auto State::ionization_enabled() const noexcept -> bool {
  return ionization_enabled_;
}

// num var accessors
auto State::n_cf() const noexcept -> int { return nvar_; }
auto State::n_pf() const noexcept -> int { return nPF_; }
auto State::n_af() const noexcept -> int { return nAF_; }
auto State::p_order() const noexcept -> int { return pOrder_; }

// view accessors
auto State::u_cf() const noexcept -> View3D<double> { return uCF_; }
auto State::u_cf_stages() const noexcept -> View4D<double> { return uCF_s_; }
auto State::u_pf() const noexcept -> View3D<double> { return uPF_; }
auto State::u_af() const noexcept -> View3D<double> { return uAF_; }
