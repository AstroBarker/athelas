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
 *inner products.
 **/

#include <iostream>
#include <algorithm> /* std::sort */
#include <math.h>    /* pow */
#include <cstdlib>   /* abs */

#include "Grid.h"
#include "LinearAlgebraModules.h"
#include "QuadratureLibrary.h"
#include "PolynomialBasis.h"
#include "Error.h"
#include "FluidUtilities.h"

/**
 * Constructor creates necessary matrices and bases, etc.
 * This has to be called after the problem is initialized.
 **/
ModalBasis::ModalBasis( Kokkos::View<double***> uPF, GridStructure& Grid,
                        unsigned int pOrder, unsigned int nN,
                        unsigned int nElements, unsigned int nGuard )
    : nX( nElements ), order( pOrder ), nNodes( nN ),
      mSize( ( nN ) * ( nN + 2 ) * ( nElements + 2 * nGuard ) ),
      MassMatrix( "MassMatrix", nElements + 2 * nGuard, pOrder ),
      Phi( "Phi", nElements + 2 * nGuard, 3 * nN + 2, pOrder ),
      dPhi( "dPhi", nElements + 2 * nGuard, 3 * nN + 2, pOrder )
{
  // --- Compute grid quantities ---
  Grid.ComputeMass( uPF );
  Grid.ComputeCenterOfMass( uPF );

  InitializeTaylorBasis( uPF, Grid );

  // InitializeLegendreBasis( uPF, Grid );
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
double ModalBasis::Taylor( unsigned int order, double eta, double eta_c )
{

  if ( order < 0 ) throw Error( "Please enter a valid polynomial order.\n" );

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
double ModalBasis::dTaylor( unsigned int order, double eta, double eta_c )
{

  if ( order < 0 ) throw Error( "Please enter a valid polynomial order.\n" );

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
double ModalBasis::InnerProduct( unsigned int m, unsigned int n,
                                 unsigned int iX, double eta_c,
                                 Kokkos::View<double***> uPF,
                                 GridStructure& Grid )
{
  double result = 0.0;
  double eta_q  = 0.0;
  double X      = 0.0;

  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    eta_q = Grid.Get_Nodes( iN );
    X     = Grid.NodeCoordinate( iX, iN );
    result += Taylor( n, eta_q, eta_c ) * Phi( iX, iN + 1, m ) *
              Grid.Get_Weights( iN ) * uPF( 0, iX, iN ) *
              Grid.Get_Widths( iX ) * Grid.Get_SqrtGm( X );
  }

  return result;
}

/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < Phi_m, Phi_n >
 * <f,g> = \sum_q \rho_q f_q g_q j^0 w_q
 **/
double ModalBasis::InnerProduct( unsigned int n, unsigned int iX, double eta_c,
                                 Kokkos::View<double***> uPF,
                                 GridStructure& Grid )
{
  double result = 0.0;
  double X      = 0.0;

  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    X = Grid.NodeCoordinate( iX, iN );
    result += Phi( iX, iN + 1, n ) * Phi( iX, iN + 1, n ) *
              Grid.Get_Weights( iN ) * uPF( 0, iX, iN ) *
              Grid.Get_Widths( iX ) * Grid.Get_SqrtGm( X );
  }

  return result;
}

// Gram-Schmidt orthogonalization to Taylor basis
// TODO: OrthoTaylor: Clean up derivative options?
double ModalBasis::OrthoTaylor( unsigned int order, unsigned int iX,
                                unsigned int i_eta, double eta, double eta_c,
                                Kokkos::View<double***> uPF,
                                GridStructure& Grid, bool derivative_option )
{

  double result      = 0.0;
  double phi_n       = 0.0;
  double numerator   = 0.0;
  double denominator = 0.0;

  // TODO: Can this be cleaned up?
  if ( not derivative_option )
  {
    result = Taylor( order, eta, eta_c );
  }
  else
  {
    result = dTaylor( order, eta, eta_c );
  }

  // if ( order == 0 ) return result;

  for ( unsigned int k = 0; k < order; k++ )
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
 * Pre-compute the orthogonal Taylor basis terms. Phi(k,eta) will store
 * the expansion terms for each order k, stored at various points eta.
 * We store: (-0.5, {GL nodes}, 0.5) for a total of nNodes+2
 **/
void ModalBasis::InitializeTaylorBasis( Kokkos::View<double***> uPF,
                                        GridStructure& Grid )
{
  const unsigned int n_eta = 3 * nNodes + 2;
  const unsigned int ilo   = Grid.Get_ilo( );
  const unsigned int ihi   = Grid.Get_ihi( );

  double eta_c;

  double eta = 0.5;
  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    eta_c = Grid.Get_CenterOfMass( iX );
    for ( unsigned int k = 0; k < order; k++ )
      for ( unsigned int i_eta = 0; i_eta < n_eta; i_eta++ )
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
          eta = Grid.Get_Nodes( i_eta - 1 );
        }
        else if ( i_eta > nNodes + 1 &&
                  i_eta <= 2 * nNodes + 1 ) // GL nodes left neighbor
        {
          eta = Grid.Get_Nodes( i_eta - nNodes - 2 ) - 1.0;
        }
        else
        {
          eta = Grid.Get_Nodes( i_eta - 2 * nNodes - 2 ) + 1.0;
        }

        Phi( iX, i_eta, k ) =
            OrthoTaylor( k, iX, i_eta, eta, eta_c, uPF, Grid, false );
        dPhi( iX, i_eta, k ) =
            OrthoTaylor( k, iX, i_eta, eta, eta_c, uPF, Grid, true );
      }
  }
  CheckOrthogonality( uPF, Grid );
  ComputeMassMatrix( uPF, Grid );

  // === Fill Guard cells ===

  // ? Using identical basis in guard cells as boundaries ?
  for ( unsigned int iX = 0; iX < ilo; iX++ )
    for ( unsigned int i_eta = 0; i_eta < n_eta; i_eta++ )
      for ( unsigned int k = 0; k < order; k++ )
      {
        Phi( ilo - 1 - iX, i_eta, k ) = Phi( ilo + iX, i_eta, k );
        Phi( ihi + 1 + iX, i_eta, k ) = Phi( ihi - iX, i_eta, k );

        dPhi( ilo - 1 - iX, i_eta, k ) = dPhi( ilo + iX, i_eta, k );
        dPhi( ihi + 1 + iX, i_eta, k ) = dPhi( ihi - iX, i_eta, k );
      }

  for ( unsigned int iX = 0; iX < ilo; iX++ )
    for ( unsigned int k = 0; k < order; k++ )
    {
      MassMatrix( ilo - 1 - iX, k ) = MassMatrix( ilo + iX, k );
      MassMatrix( ihi + 1 + iX, k ) = MassMatrix( ihi - iX, k );
    }
}

/**
 * Pre-compute the orthogonal Taylor basis terms. Phi(k,eta) will store
 * the expansion terms for each order k, stored at various points eta.
 * We store: (-0.5, {GL nodes}, 0.5) for a total of nNodes+2
 **/
void ModalBasis::InitializeLegendreBasis( Kokkos::View<double***> uPF,
                                          GridStructure& Grid )
{
  const unsigned int n_eta = 3 * nNodes + 2;
  const unsigned int ilo   = Grid.Get_ilo( );
  const unsigned int ihi   = Grid.Get_ihi( );

  double eta = 0.5;
  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    for ( unsigned int k = 0; k < order; k++ )
      for ( unsigned int i_eta = 0; i_eta < n_eta; i_eta++ )
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
          eta = Grid.Get_Nodes( i_eta - 1 );
        }
        else if ( i_eta > nNodes + 1 &&
                  i_eta < 2 * nNodes + 1 ) // GL nodes left neighbor
        {
          eta = Grid.Get_Nodes( i_eta - nNodes - 1 ) - 1.0;
        }
        else
        {
          eta = Grid.Get_Nodes( i_eta - 2 * nNodes - 1 ) + 1.0;
        }

        Phi( iX, i_eta, k )  = Legendre( k, eta );
        dPhi( iX, i_eta, k ) = dLegendre( k, eta );
      }
  }
  CheckOrthogonality( uPF, Grid );
  ComputeMassMatrix( uPF, Grid );

  // === Fill Guard cells ===

  // ? Using identical basis in guard cells as boundaries ?
  for ( unsigned int iX = 0; iX < ilo; iX++ )
    for ( unsigned int i_eta = 0; i_eta < n_eta; i_eta++ )
      for ( unsigned int k = 0; k < order; k++ )
      {
        Phi( ilo - 1 - iX, i_eta, k ) = Phi( ilo + iX, i_eta, k );
        Phi( ihi + 1 + iX, i_eta, k ) = Phi( ihi - iX, i_eta, k );

        dPhi( ilo - 1 - iX, i_eta, k ) = dPhi( ilo + iX, i_eta, k );
        dPhi( ihi + 1 + iX, i_eta, k ) = dPhi( ihi - iX, i_eta, k );
      }

  for ( unsigned int iX = 0; iX < ilo; iX++ )
    for ( unsigned int k = 0; k < order; k++ )
    {
      MassMatrix( ilo - 1 - iX, k ) = MassMatrix( ilo + iX, k );
      MassMatrix( ihi + 1 + iX, k ) = MassMatrix( ihi - iX, k );
    }
}

/**
 * The following checks orthogonality of basis functions on each cell.
 * Returns error if orthogonality is not met.
 **/
void ModalBasis::CheckOrthogonality( Kokkos::View<double***> uPF,
                                     GridStructure& Grid )
{

  const unsigned int ilo = Grid.Get_ilo( );
  const unsigned int ihi = Grid.Get_ihi( );
  double X               = 0.0;

  double result = 0.0;
  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
    for ( unsigned int k1 = 0; k1 < order; k1++ )
      for ( unsigned int k2 = 0; k2 < order; k2++ )
      {
        result = 0.0;
        for ( unsigned int i_eta = 1; i_eta <= nNodes;
              i_eta++ ) // loop over quadratures
        {
          X = Grid.NodeCoordinate( iX, i_eta - 1 );
          // Not using an InnerProduct function because their API is odd..
          result += Phi( iX, i_eta, k1 ) * Phi( iX, i_eta, k2 ) *
                    uPF( 0, iX, i_eta - 1 ) * Grid.Get_Weights( i_eta - 1 ) *
                    Grid.Get_Widths( iX ) * Grid.Get_SqrtGm( X );
        }

        if ( k1 == k2 && result == 0.0 )
        {
          throw Error( "Basis not orthogonal: Diagonal term equal to zero.\n" );
        }
        if ( k1 != k2 && std::abs( result ) > 1e-10 )
        {
          std::printf( "%d %d %.3e \n", k1, k2, result );
          throw Error( "Basis not orthogonal: Off diagonal term non-zero.\n" );
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
void ModalBasis::ComputeMassMatrix( Kokkos::View<double***> uPF,
                                    GridStructure& Grid )
{
  const unsigned int ilo    = Grid.Get_ilo( );
  const unsigned int ihi    = Grid.Get_ihi( );
  const unsigned int nNodes = Grid.Get_nNodes( );

  double result = 0.0;
  double X      = 0.0;

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    for ( unsigned int k = 0; k < order; k++ )
    {
      result = 0.0;
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        X = Grid.NodeCoordinate( iX, iN );
        result += Phi( iX, iN + 1, k ) * Phi( iX, iN + 1, k ) *
                  Grid.Get_Weights( iN ) * Grid.Get_Widths( iX ) *
                  Grid.Get_SqrtGm( X ) * uPF( 0, iX, iN );
      }
      MassMatrix( iX, k ) = result;
    }
  }
}

/**
 * Evaluate (modal) basis on element iX for quantity iCF.
 * If DerivativeOption is true, evaluate the derivative.
 **/
double ModalBasis::BasisEval( Kokkos::View<double***> U, unsigned int iX,
                              unsigned int iCF, unsigned int i_eta,
                              bool DerivativeOption ) const
{
  double result = 0.0;
  if ( DerivativeOption )
  {
    for ( unsigned int k = 0; k < order; k++ )
    {
      result += dPhi( iX, i_eta, k ) * U( iCF, iX, k );
    }
  }
  else
  {
    for ( unsigned int k = 0; k < order; k++ )
    {
      result += Phi( iX, i_eta, k ) * U( iCF, iX, k );
    }
  }
  return result;
}

// Accessor for Phi
double ModalBasis::Get_Phi( unsigned int iX, unsigned int i_eta,
                            unsigned int k ) const
{
  return Phi( iX, i_eta, k );
}

// Accessor for dPhi
double ModalBasis::Get_dPhi( unsigned int iX, unsigned int i_eta,
                             unsigned int k ) const
{
  return dPhi( iX, i_eta, k );
}

// Accessor for mass matrix
double ModalBasis::Get_MassMatrix( unsigned int iX, unsigned int k ) const
{
  return MassMatrix( iX, k );
}

// Accessor for Order
int ModalBasis::Get_Order( ) const { return order; }

// --- Legendre Methods ---

// Legendre polynomials
double ModalBasis::Legendre( unsigned int order, double x )
{

  if ( order == 0 )
  {
    return 1.0;
  }
  else if ( order == 1 )
  {
    return 2.0 * x;
  }
  else
  {

    x *= 2.0; // This maps to interval [-0.5, 0.5]

    double Pn, Pnm1; // P_n, P_{n-1}
    double Pnp1 = 0.0;

    Pnm1 = 1.0; // P_0
    Pn   = x;   //  P_1
    for ( unsigned int i = 1; i < order; i++ )
    {
      Pnp1 = 2.0 * x * Pn - Pnm1 - ( x * Pn - Pnm1 ) / ( i + 1 );
      // Pnp1 = ( (2*i + 1) * x * Pn - i * Pnm1 ) / (i + 1);

      Pnm1 = Pn;
      Pn   = Pnp1;
    }

    return Pn;
  }
}

// Derivative of Legendre polynomials
double ModalBasis::dLegendre( unsigned int order, double x )
{

  double dPn; // P_n
  // double dPnp1 = 0.0;

  dPn = 0.0;
  for ( unsigned int i = 0; i < order; i++ )
  {
    dPn = ( i + 1 ) * Legendre( i, x ) + 2.0 * x * dPn;
  }

  return dPn;
}
