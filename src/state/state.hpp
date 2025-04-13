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

  [[nodiscard]] auto Get_nCF( ) const -> int;
  [[nodiscard]] auto Get_nCR( ) const -> int;
  [[nodiscard]] auto Get_nPF( ) const -> int;
  [[nodiscard]] auto Get_nAF( ) const -> int;
  [[nodiscard]] auto Get_pOrder( ) const -> int;

  [[nodiscard]] View3D<Real> Get_uCF( ) const;
  [[nodiscard]] View3D<Real> Get_uPF( ) const;
  [[nodiscard]] View3D<Real> Get_uAF( ) const;
  [[nodiscard]] View3D<Real> Get_uCR( ) const;

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
