#ifndef STATE_HPP_
#define STATE_HPP_
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
  State( int nCF_, int nCR_, int nPF_, int nAF_, int nX_, int nG_, int nNodes_,
         int pOrder_ );

  [[nodiscard]] auto get_n_cf( ) const -> int;
  [[nodiscard]] auto get_n_cr( ) const -> int;
  [[nodiscard]] auto get_n_pf( ) const -> int;
  [[nodiscard]] auto get_n_af( ) const -> int;
  [[nodiscard]] auto get_p_order( ) const -> int;

  [[nodiscard]] View3D<Real> get_u_cf( ) const;
  [[nodiscard]] View3D<Real> get_u_pf( ) const;
  [[nodiscard]] View3D<Real> get_u_af( ) const;
  [[nodiscard]] View3D<Real> get_u_cr( ) const;

 private:
  int nCF;
  int nCR;
  int nPF;
  int nAF;
  int pOrder;

  View3D<Real> uCF{ }; // Conserved fluid
  View3D<Real> uPF{ }; // primitive fluid
  View3D<Real> uAF{ }; // auxiliary fluid
  View3D<Real> uCR{ }; // conserved radiation
};

#endif // STATE_HPP_
