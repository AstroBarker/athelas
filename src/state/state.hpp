#ifndef STATE_HPP_
#define STATE_HPP_

/**
 * File     :  state.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the state data.
 *
 **/

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Error.hpp"


class State
{
 public:
  State( const int nCF_, const int nCR_, const int nPF_, const int nAF_,
         const int nX_, const int nG_, const int nNodes_, const int pOrder_ );

  int Get_nCF( ) const;
  int Get_nCR( ) const;
  int Get_nPF( ) const;
  int Get_nAF( ) const;

  Kokkos::View<Real ***> Get_uCF( ) const;
  Kokkos::View<Real ***> Get_uPF( ) const;
  Kokkos::View<Real ***> Get_uAF( ) const;
  Kokkos::View<Real ***> Get_uCR( ) const;

 private:
  int nCF;
  int nCR;
  int nPF;
  int nAF;

  Kokkos::View<Real ***> uCF; // Conserved fluid
  Kokkos::View<Real ***> uPF; // primitive fluid
  Kokkos::View<Real ***> uAF; // auxiliary fluid
  Kokkos::View<Real ***> uCR; // conserved radiation
};

#endif // STATE_HPP_
