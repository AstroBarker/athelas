/**
 * @file slope_limiter_tvdminmod.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief TVB Minmod slope limiter for discontinuous Galerkin methods
 *
 * @details This file implements the Total Variation Diminishing (TVD) Minmod
 *          slope limiter based on the work of Cockburn & Shu. The limiter
 *          provides a robust, first-order accurate approach to preventing
 *          oscillations in discontinuous solutions.
 */

#include <algorithm> /* std::min, std::max */
#include <cstdlib> /* abs */

#include "characteristic_decomposition.hpp"
#include "grid.hpp"
#include "linear_algebra.hpp"
#include "polynomial_basis.hpp"
#include "slope_limiter.hpp"
#include "slope_limiter_utilities.hpp"

using namespace limiter_utilities;

/**
 * TVD Minmod limiter. See the Cockburn & Shu papers
 **/
void TVDMinmod::apply_slope_limiter(View3D<double> U, const GridStructure* grid,
                                    const ModalBasis* basis, const EOS* eos) {

  // Do not apply for first order method or if we don't want to.
  if (order_ == 1 || !do_limiter_) {
    return;
  }

  constexpr static double sl_threshold_ =
      1.0e-6; // TODO(astrobarker): move to input deck
  constexpr static double EPS = 1.0e-10;

  static constexpr int ilo = 1;
  static const int& ihi = grid->get_ihi();

  const int nvars = nvars_;

  Kokkos::parallel_for(
      "SlopeLimiter :: Minmod :: Reset limiter indicator",
      Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix) { limited_cell_(ix) = 0; });

  // --- Apply troubled cell indicator ---
  if (tci_opt_) {
    detect_troubled_cells(U, D_, grid, basis, vars_);
  }

  // TODO(astrobarker): this is repeated code: clean up somehow
  // --- map to characteristic vars ---
  if (characteristic_) {
    Kokkos::parallel_for(
        "SlopeLimiter :: Minmod :: ToCharacteristic",
        Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_CLASS_LAMBDA(const int ix) {
          // --- Characteristic Limiting Matrices ---
          // Note: using cell averages
          for (int iC = 0; iC < nvars; ++iC) {
            mult_(ix, iC) = U(ix, 0, iC);
          }

          auto R_i = Kokkos::subview(R_, ix, Kokkos::ALL, Kokkos::ALL);
          auto R_inv_i = Kokkos::subview(R_inv_, ix, Kokkos::ALL, Kokkos::ALL);
          auto U_c_T_i = Kokkos::subview(U_c_T_, ix, Kokkos::ALL);
          auto w_c_T_i = Kokkos::subview(w_c_T_, ix, Kokkos::ALL);
          auto Mult_i = Kokkos::subview(mult_, ix, Kokkos::ALL);
          compute_characteristic_decomposition(Mult_i, R_i, R_inv_i, eos);
          for (int k = 0; k <= 1; ++k) {
            // store w_.. = invR @ U_..
            for (int iC = 0; iC < nvars; ++iC) {
              U_c_T_i(iC) = U(ix, k, iC);
              w_c_T_i(iC) = 0.0;
            }
            MAT_MUL<3>(1.0, R_inv_i, U_c_T_i, 0.0, w_c_T_i);

            for (int iC = 0; iC < nvars; ++iC) {
              U(ix, k, iC) = w_c_T_i(iC);
            } // end loop vars
          } // end loop k
        }); // par ix
  } // end map to characteristics

  Kokkos::parallel_for(
      "SlopeLimiter :: Minmod", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix) {
        limited_cell_(ix) = 0;

        // Do nothing we don't need to limit slopes
        if (D_(ix) > tci_val_ || !tci_opt_) {
          for (int iC : vars_) {

            // --- Begin TVD Minmod Limiter --- //
            const double s_i = U(ix, 1, iC); // target cell slope
            const double c_i = U(ix, 0, iC); // target cell avg
            const double c_p = U(ix + 1, 0, iC); // cell ix + 1 avg
            const double c_m = U(ix - 1, 0, iC); // cell ix - 1 avg
            const double dx = grid->get_widths(ix);
            const double new_slope = MINMOD_B(s_i, b_tvd_ * (c_p - c_i),
                                              b_tvd_ * (c_i - c_m), dx, m_tvb_);

            // check limited slope difference vs threshold
            if (std::abs(new_slope - s_i) >
                sl_threshold_ * std::max(std::abs(s_i), EPS)) {
              // limit
              U(ix, 1, iC) = new_slope;
              // remove any higher order contributions
              for (int k = 2; k < order_; ++k) {
                U(ix, k, iC) = 0.0;
              }
            }
            // --- End TVD Minmod Limiter --- //
            // The TVDMinmod part is really small... reusing a lot of code

            // --- Note we have limited this cell --- //
            limited_cell_(ix) = 1;

          } // end loop iC
        } // end if "limit_this_cell"
      }); // par_for ix

  /* Map back to conserved variables */
  if (characteristic_) {
    Kokkos::parallel_for(
        "SlopeLimiter :: Minmod :: FromCharacteristic",
        Kokkos::RangePolicy<>(ilo, ihi + 1), KOKKOS_CLASS_LAMBDA(const int ix) {
          // --- Characteristic Limiting Matrices ---
          auto R_i = Kokkos::subview(R_, ix, Kokkos::ALL, Kokkos::ALL);
          auto U_c_T_i = Kokkos::subview(U_c_T_, ix, Kokkos::ALL);
          auto w_c_T_i = Kokkos::subview(w_c_T_, ix, Kokkos::ALL);
          for (int k = 0; k < 2; ++k) {
            // store U.. = R @ w..
            for (int iC = 0; iC < nvars; ++iC) {
              U_c_T_i(iC) = U(ix, k, iC);
              w_c_T_i(iC) = 0.0;
            }
            MAT_MUL<3>(1.0, R_i, U_c_T_i, 0.0, w_c_T_i);

            for (int iC = 0; iC < nvars; ++iC) {
              U(ix, k, iC) = w_c_T_i(iC);
            } // end loop vars
          } // end loop k
        }); // par_for ix
  } // end map from characteristics
} // end apply slope limiter

// limited_cell_ accessor
auto TVDMinmod::get_limited(const int ix) const -> int {
  return (!do_limiter_) ? 0 : limited_cell_(ix);
}

auto TVDMinmod::limited() const -> View1D<int> { return limited_cell_; }
