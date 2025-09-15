/**
 * @file slope_limiter_weno.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Implementation of the WENO-Z slope limiter for discontinuous Galerkin
 *        methods
 *
 * @details This file implements the WENO-Z slope limiter based on H. Zhu 2020,
 *          "Simple, high-order compact WENO RKDG slope limiter". The limiter
 *          uses a compact stencil approach to maintain high-order accuracy
 *          while preventing oscillations.
 */

#include "characteristic_decomposition.hpp"
#include "grid.hpp"
#include "linear_algebra.hpp"
#include "polynomial_basis.hpp"
#include "slope_limiter.hpp"
#include "slope_limiter_utilities.hpp"

using namespace limiter_utilities;

/**
 * Apply the slope limiter. We use a compact stencil WENO-Z limiter
 * H. Zhu 2020, simple, high-order compact WENO RKDG slope limiter
 **/
void WENO::apply_slope_limiter(View3D<double> U, const GridStructure *grid,
                               const ModalBasis *basis, const EOS *eos) {

  // Do not apply for first order method or if we don't want to.
  if (order_ == 1 || !do_limiter_) {
    return;
  }

  static constexpr int ilo = 1;
  static const int &ihi = grid->get_ihi();

  const auto nvars = nvars_;

  // --- Apply troubled cell indicator ---
  if (tci_opt_) {
    detect_troubled_cells(U, D_, grid, basis, vars_);
  }

  /* map to characteristic vars */
  if (characteristic_) {
    Kokkos::parallel_for(
        "SlopeLimiter :: WENO :: ToCharacteristic",
        Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_CLASS_LAMBDA(const int ix) {
          // --- Characteristic Limiting Matrices ---
          // Note: using cell averages
          for (int iC = 0; iC < nvars; iC++) {
            mult_(ix, iC) = U(ix, 0, iC);
          }

          auto R_i = Kokkos::subview(R_, ix, Kokkos::ALL, Kokkos::ALL);
          auto R_inv_i = Kokkos::subview(R_inv_, ix, Kokkos::ALL, Kokkos::ALL);
          auto U_c_T_i = Kokkos::subview(U_c_T_, ix, Kokkos::ALL);
          auto w_c_T_i = Kokkos::subview(w_c_T_, ix, Kokkos::ALL);
          auto mult_i = Kokkos::subview(mult_, ix, Kokkos::ALL);
          compute_characteristic_decomposition(mult_i, R_i, R_inv_i, eos);
          for (int k = 0; k < order_; k++) {
            // store w_.. = invR @ U_..
            for (int iC = 0; iC < nvars; iC++) {
              U_c_T_i(iC) = U(ix, k, iC);
              w_c_T_i(iC) = 0.0;
            }
            MAT_MUL<3>(1.0, R_inv_i, U_c_T_i, 1.0, w_c_T_i);

            for (int iC = 0; iC < nvars; iC++) {
              U(ix, k, iC) = w_c_T_i(iC);
            } // end loop vars
          } // end loop k
        }); // par ix
  } // end map to characteristics

  Kokkos::parallel_for(
      "SlopeLimiter :: WENO", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        limited_cell_(ix) = 0;

        // Do nothing we don't need to limit slopes
        if (D_(ix) > tci_val_ || !tci_opt_) {
          for (int iC : vars_) {
            // get scratch modified_polynomial view for this cell's work
            auto modified_polynomial_i = Kokkos::subview(
                modified_polynomial_, ix, Kokkos::ALL, Kokkos::ALL);

            // modify polynomials
            modify_polynomial(U, modified_polynomial_i, gamma_i_, gamma_l_,
                              gamma_r_, ix, iC);

            const double beta_l = smoothness_indicator(
                U, modified_polynomial_i, grid, basis, ix, 0, iC); // ix - 1
            const double beta_i = smoothness_indicator(
                U, modified_polynomial_i, grid, basis, ix, 1, iC); // ix
            const double beta_r = smoothness_indicator(
                U, modified_polynomial_i, grid, basis, ix, 2, iC); // ix + 1
            const double tau = weno_tau(beta_l, beta_i, beta_r, weno_r_);

            // nonlinear weights w
            const double dx_i = 0.1 * grid->get_widths(ix);
            double w_l = non_linear_weight(gamma_l_, beta_l, tau, dx_i);
            double w_i = non_linear_weight(gamma_i_, beta_i, tau, dx_i);
            double w_r = non_linear_weight(gamma_r_, beta_r, tau, dx_i);

            const double sum_w = w_l + w_i + w_r;
            w_l /= sum_w;
            w_i /= sum_w;
            w_r /= sum_w;

            // update solution via WENO
            for (int k = 1; k < order_; k++) {
              U(ix, k, iC) = w_l * modified_polynomial_i(0, k) +
                             w_i * modified_polynomial_i(1, k) +
                             w_r * modified_polynomial_i(2, k);
            }

            /* Note we have limited this cell */
            limited_cell_(ix) = 1;

          } // end loop iC
        } // end if "limit_this_cell"
      }); // par_for ix

  /* Map back to conserved variables */
  if (characteristic_) {
    Kokkos::parallel_for(
        "SlopeLimiter :: WENO :: FromCharacteristic",
        Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_CLASS_LAMBDA(const int ix) {
          // --- Characteristic Limiting Matrices ---
          auto R_i = Kokkos::subview(R_, Kokkos::ALL, ix, Kokkos::ALL);
          auto U_c_T_i = Kokkos::subview(U_c_T_, ix, Kokkos::ALL);
          auto w_c_T_i = Kokkos::subview(w_c_T_, ix, Kokkos::ALL);
          for (int k = 0; k < order_; k++) {
            // store w_.. = invR @ U_..
            for (int iC = 0; iC < nvars; iC++) {
              U_c_T_i(iC) = U(ix, k, iC);
              w_c_T_i(iC) = 0.0;
            }
            MAT_MUL<3>(1.0, R_i, U_c_T_i, 1.0, w_c_T_i);

            for (int iC = 0; iC < nvars; iC++) {
              U(ix, k, iC) = w_c_T_i(iC);
            } // end loop vars
          } // end loop k
        }); // par_for ix
  } // end map from characteristics
} // end apply slope limiter

// LimitedCell accessor
auto WENO::get_limited(const int ix) const -> int {
  return (!do_limiter_) ? 0 : limited_cell_(ix);
}

auto WENO::limited() const -> View1D<int> { return limited_cell_; }
