/**
 * File     :  PolynomialBasis.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Functions for polynomial basis
 * Contains : Class for Taylor basis.
 * Also  Lagrange, Legendre polynomials, arbitrary degree.
 *
 * TODO: Plenty of cleanup to be done. OrthoTaylor, handling of derivatives, and
 * inner products. A lot of nearly duplicate code.
 * Maybe much of this can be compile time, as well.
 **/

#include <iostream>
#include <algorithm> /* std::sort */
#include <math.h>    /* pow */
#include <cstdlib>   /* abs */

#include "Grid.hpp"
#include "LinearAlgebraModules.hpp"
#include "QuadratureLibrary.hpp"
#include "PolynomialBasis.hpp"
#include "Error.hpp"
#include "FluidUtilities.hpp"

/**
 * Constructor creates necessary matrices and bases, etc.
 * This has to be called after the problem is initialized.
 **/
ModalBasis::ModalBasis( Kokkos::View<Real ***> uPF, GridStructure *Grid,
                        UInt pOrder, UInt nN, UInt nElements, UInt nGuard )
    : nX( nElements ), order( pOrder ), nNodes( nN ),
      mSize( ( nN ) * ( nN + 2 ) * ( nElements + 2 * nGuard ) ),
      MassMatrix( "MassMatrix", nElements + 2 * nGuard, pOrder ),
      Phi( "Phi", nElements + 2 * nGuard, 3 * nN + 2, pOrder ),
      dPhi( "dPhi", nElements + 2 * nGuard, 3 * nN + 2, pOrder )
{
  // --- Compute grid quantities ---
  Grid->ComputeMass( uPF );
  Grid->ComputeCenterOfMass( uPF );

  //InitializeTaylorBasis( uPF, Grid );

  InitializeLegendreBasis( uPF, Grid );
}

// --- Taylor Methods ---

/**
 * Return Taylor polynomial of given order
 *
 * Parameters:
 * -----------
 * eta   : coordinate
 * eta_c : center of mass
 **/
Real ModalBasis::Taylor( UInt order, Real eta, Real eta_c )
{

  if ( order < 0 ) throw Error( "! Please enter a valid polynomial order.\n" );

  // Handle constant and linear terms separately -- no need to exponentiate.
  if ( order == 0 )
  {
    return 1.0;
  }
  else if ( order == 1 )
  {
    return eta - eta_c;
  }
  else
  {
    return std::pow( eta - eta_c, order );
  }
}

/**
 * Return derivative of Taylor polynomial of given order
 *
 * Parameters:
 * -----------
 * eta : coordinate
 * eta_c: center of mass
 **/
Real ModalBasis::dTaylor( UInt order, Real eta, Real eta_c )
{

  if ( order < 0 ) throw Error( " ! Please enter a valid polynomial order.\n" );

  // Handle first few terms separately -- no need to call std::pow
  if ( order == 0 )
  {
    return 0.0;
  }
  else if ( order == 1 )
  {
    return 1.0;
  }
  else if ( order == 2 )
  {
    return 2 * ( eta - eta_c );
  }
  else
  {
    return (order)*std::pow( eta - eta_c, order - 1 );
  }
}

/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < Psi_m, Phi_n >
 * <f,g> = \sum_q \rho_q f_Q g_q j^0 w_q
 * TODO: Make InnerProduct functions cleaner????
 **/
Real ModalBasis::InnerProduct( const UInt m, const UInt n, const UInt iX,
                               const Real eta_c,
                               const Kokkos::View<Real ***> uPF,
                               GridStructure *Grid )
{
  Real result = 0.0;
  Real eta_q  = 0.0;
  Real X      = 0.0;

  for ( UInt iN = 0; iN < nNodes; iN++ )
  {
    eta_q = Grid->Get_Nodes( iN );
    X     = Grid->NodeCoordinate( iX, iN );
    result += Taylor( n, eta_q, eta_c ) * Phi( iX, iN + 1, m ) *
              Grid->Get_Weights( iN ) * uPF( 0, iX, iN ) *
              Grid->Get_Widths( iX ) * Grid->Get_SqrtGm( X );
  }

  return result;
}

/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < Phi_m, Phi_n >
 * <f,g> = \sum_q \rho_q f_q g_q j^0 w_q
 **/
Real ModalBasis::InnerProduct( const UInt n, const UInt iX, const Real eta_c,
                               const Kokkos::View<Real ***> uPF,
                               GridStructure *Grid )
{
  Real result = 0.0;
  Real X      = 0.0;

  for ( UInt iN = 0; iN < nNodes; iN++ )
  {
    X = Grid->NodeCoordinate( iX, iN );
    result += Phi( iX, iN + 1, n ) * Phi( iX, iN + 1, n ) *
              Grid->Get_Weights( iN ) * uPF( 0, iX, iN ) *
              Grid->Get_Widths( iX ) * Grid->Get_SqrtGm( X );
  }

  return result;
}


// Gram-Schmidt orthogonalization to Legendre basis
Real ModalBasis::Ortho( Real (*f)(UInt n, Real x, Real x_c),
                        const UInt order, const UInt iX, const UInt i_eta,
                        const Real eta, const Real eta_c,
                        const Kokkos::View<Real ***> uPF,
                        GridStructure *Grid,
                        bool const derivative_option )
{

  Real result      = 0.0;
  Real phi_n       = 0.0;
  Real numerator   = 0.0;
  Real denominator = 0.0;

  // TODO: Can this be cleaned up?
  if ( not derivative_option )
  {
    result = f( order, eta, eta_c );
  }
  else
  {
    result = f( order, eta, eta_c );
  }

  // if ( order == 0 ) return result;

  for ( UInt k = 0; k < order; k++ )
  {
    numerator   = InnerProduct( order - k - 1, order, iX, eta_c, uPF, Grid );
    denominator = InnerProduct( order - k - 1, iX, eta_c, uPF, Grid );
    // ? Can this be cleaned up?
    if ( not derivative_option )
    {
      phi_n = Phi( iX, i_eta, order - k - 1 );
    }
    else
    {
      phi_n = dPhi( iX, i_eta, order - k - 1 );
    }
    result -= ( numerator / denominator ) * phi_n;
  }

  return result;
}
/**
 * Pre-compute the orthogonal Taylor basis terms. Phi(iX,k,eta) will store
 * the expansion terms for each order k, stored at various points eta.
 * We store: (-0.5, {GL nodes}, 0.5) for a total of nNodes+2
 **/
void ModalBasis::InitializeTaylorBasis( const Kokkos::View<Real ***> uPF,
                                        GridStructure *Grid )
{
  const UInt n_eta = 3 * nNodes + 2;
  const UInt ilo   = Grid->Get_ilo( );
  const UInt ihi   = Grid->Get_ihi( );

  Real eta_c;

  Real eta = 0.5;
  for ( UInt iX = ilo; iX <= ihi; iX++ )
  {
    eta_c = Grid->Get_CenterOfMass( iX );
    for ( UInt k = 0; k < order; k++ )
      for ( UInt i_eta = 0; i_eta < n_eta; i_eta++ )
      {
        // face values
        if ( i_eta == 0 )
        {
          eta = -0.5;
        }
        else if ( i_eta == nNodes + 1 )
        {
          eta = +0.5;
        }
        else if ( i_eta > 0 && i_eta < nNodes + 1 ) // GL nodes
        {
          eta = Grid->Get_Nodes( i_eta - 1 );
        }
        else if ( i_eta > nNodes + 1 &&
                  i_eta <= 2 * nNodes + 1 ) // GL nodes left neighbor
        {
          eta = Grid->Get_Nodes( i_eta - nNodes - 2 ) - 1.0;
        }
        else
        {
          eta = Grid->Get_Nodes( i_eta - 2 * nNodes - 2 ) + 1.0;
        }

        Phi( iX, i_eta, k ) =
            Ortho( Taylor, k, iX, i_eta, eta, eta_c, uPF, Grid, false );
        dPhi( iX, i_eta, k ) =
            Ortho( dTaylor, k, iX, i_eta, eta, eta_c, uPF, Grid, true );
      }
  }
  CheckOrthogonality( uPF, Grid );
  ComputeMassMatrix( uPF, Grid );

  // === Fill Guard cells ===

  // ? Using identical basis in guard cells as boundaries ?
  for ( UInt iX = 0; iX < ilo; iX++ )
    for ( UInt i_eta = 0; i_eta < n_eta; i_eta++ )
      for ( UInt k = 0; k < order; k++ )
      {
        Phi( ilo - 1 - iX, i_eta, k ) = Phi( ilo + iX, i_eta, k );
        Phi( ihi + 1 + iX, i_eta, k ) = Phi( ihi - iX, i_eta, k );

        dPhi( ilo - 1 - iX, i_eta, k ) = dPhi( ilo + iX, i_eta, k );
        dPhi( ihi + 1 + iX, i_eta, k ) = dPhi( ihi - iX, i_eta, k );
      }

  for ( UInt iX = 0; iX < ilo; iX++ )
    for ( UInt k = 0; k < order; k++ )
    {
      MassMatrix( ilo - 1 - iX, k ) = MassMatrix( ilo + iX, k );
      MassMatrix( ihi + 1 + iX, k ) = MassMatrix( ihi - iX, k );
    }
}

/**
 * Pre-compute the orthogonal Taylor basis terms. Phi(iX,k,eta) will store
 * the expansion terms for each order k, stored at various points eta.
 * We store: (-0.5, {GL nodes}, 0.5) for a total of nNodes+2
 * TODO: Incorporate COM centering?
 **/
void ModalBasis::InitializeLegendreBasis( const Kokkos::View<Real ***> uPF,
                                          GridStructure *Grid )
{
  const UInt n_eta = 3 * nNodes + 2;
  const UInt ilo   = Grid->Get_ilo( );
  const UInt ihi   = Grid->Get_ihi( );

  Real eta = 0.5;
  for ( UInt iX = ilo; iX <= ihi; iX++ )
  {
    for ( UInt k = 0; k < order; k++ )
      for ( UInt i_eta = 0; i_eta < n_eta; i_eta++ )
      {
        // face values
        if ( i_eta == 0 )
        {
          eta = -0.5;
        }
        else if ( i_eta == nNodes + 1 )
        {
          eta = +0.5;
        }
        else if ( i_eta > 0 && i_eta < nNodes + 1 ) // GL nodes
        {
          eta = Grid->Get_Nodes( i_eta - 1 );
        }
        else if ( i_eta > nNodes + 1 &&
                  i_eta < 2 * nNodes + 1 ) // GL nodes left neighbor
        {
          eta = Grid->Get_Nodes( i_eta - nNodes - 1 ) - 1.0;
        }
        else
        {
          eta = Grid->Get_Nodes( i_eta - 2 * nNodes - 1 ) + 1.0;
        }

        Phi( iX, i_eta, k ) =
            Ortho( Legendre, k, iX, i_eta, eta, 0.0, uPF, Grid, false );
        dPhi( iX, i_eta, k ) =
            Ortho( dLegendre, k, iX, i_eta, eta, 0.0, uPF, Grid, true );
      }
  }
  CheckOrthogonality( uPF, Grid );
  ComputeMassMatrix( uPF, Grid );

  // === Fill Guard cells ===

  // ? Using identical basis in guard cells as boundaries ?
  for ( UInt iX = 0; iX < ilo; iX++ )
    for ( UInt i_eta = 0; i_eta < n_eta; i_eta++ )
      for ( UInt k = 0; k < order; k++ )
      {
        Phi( ilo - 1 - iX, i_eta, k ) = Phi( ilo + iX, i_eta, k );
        Phi( ihi + 1 + iX, i_eta, k ) = Phi( ihi - iX, i_eta, k );

        dPhi( ilo - 1 - iX, i_eta, k ) = dPhi( ilo + iX, i_eta, k );
        dPhi( ihi + 1 + iX, i_eta, k ) = dPhi( ihi - iX, i_eta, k );
      }

  for ( UInt iX = 0; iX < ilo; iX++ )
    for ( UInt k = 0; k < order; k++ )
    {
      MassMatrix( ilo - 1 - iX, k ) = MassMatrix( ilo + iX, k );
      MassMatrix( ihi + 1 + iX, k ) = MassMatrix( ihi - iX, k );
    }
}

/**
 * The following checks orthogonality of basis functions on each cell.
 * Returns error if orthogonality is not met.
 **/
void ModalBasis::CheckOrthogonality( const Kokkos::View<Real ***> uPF,
                                     GridStructure *Grid )
{

  const UInt ilo = Grid->Get_ilo( );
  const UInt ihi = Grid->Get_ihi( );
  Real X         = 0.0;

  Real result = 0.0;
  for ( UInt iX = ilo; iX <= ihi; iX++ )
    for ( UInt k1 = 0; k1 < order; k1++ )
      for ( UInt k2 = 0; k2 < order; k2++ )
      {
        result = 0.0;
        for ( UInt i_eta = 1; i_eta <= nNodes;
              i_eta++ ) // loop over quadratures
        {
          X = Grid->NodeCoordinate( iX, i_eta - 1 );
          // Not using an InnerProduct function because their API is odd..
          result += Phi( iX, i_eta, k1 ) * Phi( iX, i_eta, k2 ) *
                    uPF( 0, iX, i_eta - 1 ) * Grid->Get_Weights( i_eta - 1 ) *
                    Grid->Get_Widths( iX ) * Grid->Get_SqrtGm( X );
        }

        if ( k1 == k2 && result == 0.0 )
        {
          throw Error(
              " ! Basis not orthogonal: Diagonal term equal to zero.\n" );
        }
        if ( k1 != k2 && std::abs( result ) > 1e-10 )
        {
          std::printf( "%d %d %.3e \n", k1, k2, result );
          throw Error(
              " ! Basis not orthogonal: Off diagonal term non-zero.\n" );
        }
      }
}

/**
 * Computes \int \rho \phi_m \phi_m dw on each cell
 * TODO: Extend mass matrix to more nodes
 * ? Do I need more integration nodes for the mass matrix? ?
 * ? If so, how do I expand this ?
 * ? I would need to compute and store more GL nodes, weights ?
 **/
void ModalBasis::ComputeMassMatrix( const Kokkos::View<Real ***> uPF,
                                    GridStructure *Grid )
{
  const UInt ilo    = Grid->Get_ilo( );
  const UInt ihi    = Grid->Get_ihi( );
  const UInt nNodes = Grid->Get_nNodes( );

  Real result = 0.0;
  Real X      = 0.0;

  for ( UInt iX = ilo; iX <= ihi; iX++ )
  {
    for ( UInt k = 0; k < order; k++ )
    {
      result = 0.0;
      for ( UInt iN = 0; iN < nNodes; iN++ )
      {
        X = Grid->NodeCoordinate( iX, iN );
        result += Phi( iX, iN + 1, k ) * Phi( iX, iN + 1, k ) *
                  Grid->Get_Weights( iN ) * Grid->Get_Widths( iX ) *
                  Grid->Get_SqrtGm( X ) * uPF( 0, iX, iN );
      }
      MassMatrix( iX, k ) = result;
    }
  }
}

/**
 * Evaluate (modal) basis on element iX for quantity iCF.
 * If DerivativeOption is true, evaluate the derivative.
 **/
Real ModalBasis::BasisEval( Kokkos::View<Real ***> U, const UInt iX,
                            const UInt iCF, const UInt i_eta,
                            const bool DerivativeOption ) const
{
  Real result = 0.0;
  if ( DerivativeOption )
  {
    for ( UInt k = 0; k < order; k++ )
    {
      result += dPhi( iX, i_eta, k ) * U( iCF, iX, k );
    }
  }
  else
  {
    for ( UInt k = 0; k < order; k++ )
    {
      result += Phi( iX, i_eta, k ) * U( iCF, iX, k );
    }
  }
  return result;
}

// Accessor for Phi
Real ModalBasis::Get_Phi( UInt iX, UInt i_eta, UInt k ) const
{
  return Phi( iX, i_eta, k );
}

// Accessor for dPhi
Real ModalBasis::Get_dPhi( UInt iX, UInt i_eta, UInt k ) const
{
  return dPhi( iX, i_eta, k );
}

// Accessor for mass matrix
Real ModalBasis::Get_MassMatrix( UInt iX, UInt k ) const
{
  return MassMatrix( iX, k );
}

// Accessor for Order
int ModalBasis::Get_Order( ) const { return order; }

// --- Legendre Methods ---
/* TODO: Make sure that x_c offset for Legendre works with COM != 0 */

Real ModalBasis::Legendre( UInt n, Real x, Real x_c )
{
  return Legendre( n, x - x_c );
}

Real ModalBasis::dLegendre( UInt n, Real x, Real x_c )
{
  return dLegendre( n, x - x_c );
}

// Legendre polynomials
Real ModalBasis::Legendre( UInt n, Real x )
{
  return ( n == 0 )   ? 1.0
         : ( n == 1 ) ? x
                      : ( ( ( 2 * n ) - 1 ) * x * Legendre( n - 1, x ) -
                          ( n - 1 ) * Legendre( n - 2, x ) ) /
                            n;
}

// Derivative of Legendre polynomials
Real ModalBasis::dLegendre( UInt order, Real x )
{

  Real dPn; // P_n
  // Real dPnp1 = 0.0;

  dPn = 0.0;
  for ( UInt i = 0; i < order; i++ )
  {
    dPn = ( i + 1 ) * Legendre( i, x ) +  x * dPn;
  }

  return dPn;
}
