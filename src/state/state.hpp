#pragma once
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

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"

class State {
 public:
  State( int nCF, int nCR, int nPF, int nAF, int nX_, int nG_, int nNodes_,
         int pOrder );

  [[nodiscard]] auto get_n_cf( ) const noexcept -> int;
  [[nodiscard]] auto get_n_cr( ) const noexcept -> int;
  [[nodiscard]] auto get_n_pf( ) const noexcept -> int;
  [[nodiscard]] auto get_n_af( ) const noexcept -> int;
  [[nodiscard]] auto get_p_order( ) const noexcept -> int;

  [[nodiscard]] auto get_u_cf( ) const noexcept -> View3D<Real>;
  [[nodiscard]] auto get_u_pf( ) const noexcept -> View3D<Real>;
  [[nodiscard]] auto get_u_af( ) const noexcept -> View3D<Real>;
  [[nodiscard]] auto get_u_cr( ) const noexcept -> View3D<Real>;

 private:
  int nCF_;
  int nCR_;
  int nPF_;
  int nAF_;
  int pOrder_;

  View3D<Real> uCF_{ }; // Conserved fluid
  View3D<Real> uPF_{ }; // primitive fluid
  View3D<Real> uAF_{ }; // auxiliary fluid
  View3D<Real> uCR_{ }; // conserved radiation
};
