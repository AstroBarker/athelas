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

#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "limiters/characteristic_decomposition.hpp"
#include "limiters/slope_limiter.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "linalg/linear_algebra.hpp"
#include "loop_layout.hpp"

namespace athelas {

using basis::ModalBasis;
using eos::EOS;

/**
 * TVD Minmod limiter. See the Cockburn & Shu papers
 **/
void TVDMinmod::apply_slope_limiter(AthelasArray3D<double> U,
                                    const GridStructure *grid,
                                    const ModalBasis *basis, const EOS *eos) {

  // Do not apply for first order method or if we don't want to.
  if (order_ == 1 || !do_limiter_) {
    return;
  }

  constexpr static double sl_threshold_ =
      1.0e-6; // TODO(astrobarker): move to input deck
  constexpr static double EPS = 1.0e-10;

  static constexpr int ilo = 1;
  static const int &ihi = grid->get_ihi();

  const int nvars = nvars_;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: Minmod :: Reset indicator",
      DevExecSpace(), ilo, ihi,
      KOKKOS_CLASS_LAMBDA(const int i) { limited_cell_(i) = 0; });

  // --- Apply troubled cell indicator ---
  if (tci_opt_) {
    detect_troubled_cells(U, D_, grid, basis, vars_);
  }

  // TODO(astrobarker): this is repeated code: clean up somehow
  // --- map to characteristic vars ---
  if (characteristic_) {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: Minmod :: ToCharacteristic",
        DevExecSpace(), ilo, ihi, KOKKOS_CLASS_LAMBDA(const int i) {
          // --- Characteristic Limiting Matrices ---
          // Note: using cell averages
          for (int v = 0; v < nvars; ++v) {
            mult_(i, v) = U(i, 0, v);
          }

          auto R_i = Kokkos::subview(R_, i, Kokkos::ALL, Kokkos::ALL);
          auto R_inv_i = Kokkos::subview(R_inv_, i, Kokkos::ALL, Kokkos::ALL);
          auto U_c_T_i = Kokkos::subview(U_c_T_, i, Kokkos::ALL);
          auto w_c_T_i = Kokkos::subview(w_c_T_, i, Kokkos::ALL);
          auto Mult_i = Kokkos::subview(mult_, i, Kokkos::ALL);
          compute_characteristic_decomposition(Mult_i, R_i, R_inv_i, eos);
          for (int k = 0; k <= 1; ++k) {
            // store w_.. = invR @ U_..
            for (int v = 0; v < nvars; ++v) {
              U_c_T_i(v) = U(i, k, v);
              w_c_T_i(v) = 0.0;
            }
            MAT_MUL<3>(1.0, R_inv_i, U_c_T_i, 0.0, w_c_T_i);

            for (int v = 0; v < nvars; ++v) {
              U(i, k, v) = w_c_T_i(v);
            } // end loop vars
          } // end loop k
        }); // par i
  } // end map to characteristics

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: Minmod", DevExecSpace(), ilo,
      ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        limited_cell_(i) = 0;

        // Do nothing we don't need to limit slopes
        if (D_(i) > tci_val_ || !tci_opt_) {
          for (int v : vars_) {

            // --- Begin TVD Minmod Limiter --- //
            const double s_i = U(i, 1, v); // target cell slope
            const double c_i = U(i, 0, v); // target cell avg
            const double c_p = U(i + 1, 0, v); // cell i + 1 avg
            const double c_m = U(i - 1, 0, v); // cell i - 1 avg
            const double dx = grid->get_widths(i);
            const double new_slope = MINMOD_B(s_i, b_tvd_ * (c_p - c_i),
                                              b_tvd_ * (c_i - c_m), dx, m_tvb_);

            // check limited slope difference vs threshold
            if (std::abs(new_slope - s_i) >
                sl_threshold_ * std::max(std::abs(s_i), EPS)) {
              // limit
              U(i, 1, v) = new_slope;
              // remove any higher order contributions
              for (int k = 2; k < order_; ++k) {
                U(i, k, v) = 0.0;
              }
            }
            // --- End TVD Minmod Limiter --- //
            // The TVDMinmod part is really small... reusing a lot of code

            // --- Note we have limited this cell --- //
            limited_cell_(i) = 1;

          } // end loop v
        } // end if "limit_this_cell"
      }); // par_for i

  /* Map back to conserved variables */
  if (characteristic_) {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN,
        "SlopeLimiter :: Minmod :: FromCharacteristic", DevExecSpace(), ilo,
        ihi, KOKKOS_CLASS_LAMBDA(const int i) {
          // --- Characteristic Limiting Matrices ---
          auto R_i = Kokkos::subview(R_, i, Kokkos::ALL, Kokkos::ALL);
          auto U_c_T_i = Kokkos::subview(U_c_T_, i, Kokkos::ALL);
          auto w_c_T_i = Kokkos::subview(w_c_T_, i, Kokkos::ALL);
          for (int k = 0; k < 2; ++k) {
            // store U.. = R @ w..
            for (int v = 0; v < nvars; ++v) {
              U_c_T_i(v) = U(i, k, v);
              w_c_T_i(v) = 0.0;
            }
            MAT_MUL<3>(1.0, R_i, U_c_T_i, 0.0, w_c_T_i);

            for (int v = 0; v < nvars; ++v) {
              U(i, k, v) = w_c_T_i(v);
            } // end loop vars
          } // end loop k
        }); // par_for i
  } // end map from characteristics
} // end apply slope limiter

// limited_cell_ accessor
auto TVDMinmod::get_limited(const int i) const -> int {
  return (!do_limiter_) ? 0 : limited_cell_(i);
}

auto TVDMinmod::limited() const -> AthelasArray1D<int> { return limited_cell_; }
} // namespace athelas
