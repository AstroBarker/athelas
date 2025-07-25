#pragma once
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
 *
 * TODO(astrobarker): clean up nvars / vars business
 */

#include <variant>

#include "abstractions.hpp"
#include "eos/eos_variant.hpp"
#include "slope_limiter_base.hpp"

class WENO : public SlopeLimiterBase<WENO> {
 public:
  WENO() = default;
  WENO(const GridStructure* grid, const ProblemIn* pin,
       const std::vector<int>& vars, const int nvars)
      : do_limiter_(pin->do_limiter), order_(pin->pOrder), nvars_(nvars),
        gamma_i_(pin->gamma_i), gamma_l_(pin->gamma_l), gamma_r_(pin->gamma_r),
        weno_r_(pin->weno_r), characteristic_(pin->Characteristic),
        tci_opt_(pin->TCI_Option), tci_val_(pin->TCI_Threshold), vars_(vars),
        modified_polynomial_("modified_polynomial", grid->get_n_elements() + 2,
                             nvars, pin->pOrder),
        R_("R Matrix", nvars, nvars, grid->get_n_elements() + 2),
        R_inv_("invR Matrix", nvars, nvars, grid->get_n_elements() + 2),
        U_c_T_("U_c_T", nvars, grid->get_n_elements() + 2),
        w_c_T_("w_c_T", nvars, grid->get_n_elements() + 2),
        mult_("Mult", nvars, grid->get_n_elements() + 2),
        D_("TCI", nvars, grid->get_n_elements() + 2),
        limited_cell_("LimitedCell", grid->get_n_elements() + 2) {}

  void apply_slope_limiter(View3D<double> U, const GridStructure* grid,
                           const ModalBasis* basis, const EOS* eos);
  [[nodiscard]] auto get_limited(int iX) const -> int;

 private:
  bool do_limiter_{};
  int order_{};
  int nvars_{};
  double gamma_i_{};
  double gamma_l_{};
  double gamma_r_{};
  double weno_r_{};
  bool characteristic_{};
  bool tci_opt_{};
  double tci_val_{};
  std::vector<int> vars_;

  View3D<double> modified_polynomial_{};

  View3D<double> R_{};
  View3D<double> R_inv_{};

  // --- Slope limiter quantities ---

  View2D<double> U_c_T_{};

  // characteristic forms
  View2D<double> w_c_T_{};

  // matrix mult scratch scape
  View2D<double> mult_{};

  View2D<double> D_{};
  View1D<int> limited_cell_{};
};

class TVDMinmod : public SlopeLimiterBase<TVDMinmod> {
 public:
  TVDMinmod() = default;
  TVDMinmod(const GridStructure* grid, const ProblemIn* pin,
            const std::vector<int>& vars, const int nvars)
      : do_limiter_(pin->do_limiter), order_(pin->pOrder), nvars_(nvars),
        b_tvd_(pin->b_tvd), m_tvb_(pin->m_tvb),
        characteristic_(pin->Characteristic), tci_opt_(pin->TCI_Option),
        tci_val_(pin->TCI_Threshold), vars_(vars),
        R_("R Matrix", nvars, nvars,
           grid->get_n_elements() + 2 * grid->get_guard()),
        R_inv_("invR Matrix", nvars, nvars,
               grid->get_n_elements() + 2 * grid->get_guard()),
        U_c_T_("U_c_T", nvars, grid->get_n_elements() + 2),
        w_c_T_("w_c_T", nvars, grid->get_n_elements() + 2),
        mult_("Mult", nvars, grid->get_n_elements() + 2),
        D_("TCI", nvars, grid->get_n_elements() + 2 * grid->get_guard()),
        limited_cell_("LimitedCell",
                      grid->get_n_elements() + 2 * grid->get_guard()) {}
  void apply_slope_limiter(View3D<double> U, const GridStructure* grid,
                           const ModalBasis* basis, const EOS* eos);
  [[nodiscard]] auto get_limited(int iX) const -> int;

 private:
  bool do_limiter_{};
  int order_{};
  int nvars_{};
  double b_tvd_{};
  double m_tvb_{};
  bool characteristic_{};
  bool tci_opt_{};
  double tci_val_{};
  std::vector<int> vars_;

  View3D<double> R_{};
  View3D<double> R_inv_{};

  // --- Slope limiter quantities ---

  View2D<double> U_c_T_{};

  // characteristic forms
  View2D<double> w_c_T_{};

  // matrix mult scratch scape
  View2D<double> mult_{};

  View2D<double> D_{};
  View1D<int> limited_cell_{};
};

using SlopeLimiter = std::variant<WENO, TVDMinmod>;

// std::visit functions
KOKKOS_INLINE_FUNCTION void apply_slope_limiter(SlopeLimiter* limiter,
                                                View3D<double> U,
                                                const GridStructure* grid,
                                                const ModalBasis* basis,
                                                const EOS* eos) {
  std::visit(
      [&U, &grid, &basis, &eos](auto& limiter) {
        limiter.apply_slope_limiter(U, grid, basis, eos);
      },
      *limiter);
}
KOKKOS_INLINE_FUNCTION auto get_limited(SlopeLimiter* limiter, const int iX)
    -> int {
  return std::visit([&iX](auto& limiter) { return limiter.get_limited(iX); },
                    *limiter);
}
