/**
 * @file polynomial_basis.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Core polynomial basis functions
 *
 * @details Provides means to construct and evaluate bases
 *            - legendre
 *            - taylor
 */

#include <algorithm> /* std::sort */
#include <cmath> /* pow */
#include <cstdlib> /* abs */
#include <iostream>
#include <print>

#include "abstractions.hpp"
#include "error.hpp"
#include "fluid_utilities.hpp"
#include "grid.hpp"
#include "linear_algebra.hpp"
#include "polynomial_basis.hpp"
#include "quadrature.hpp"

/**
 * Constructor creates necessary matrices and bases, etc.
 * This has to be called after the problem is initialized.
 **/
ModalBasis::ModalBasis( poly_basis::poly_basis basis, const View3D<Real> uPF,
                        GridStructure* Grid, const int pOrder, const int nN,
                        const int nElements, const int nGuard )
    : nX( nElements ), order( pOrder ), nNodes( nN ),
      mSize( ( nN ) * ( nN + 2 ) * ( nElements + 2 * nGuard ) ),
      MassMatrix( "MassMatrix", nElements + 2 * nGuard, pOrder ),
      Phi( "Phi", nElements + 2 * nGuard, 3 * nN + 2, pOrder ),
      dPhi( "dPhi", nElements + 2 * nGuard, 3 * nN + 2, pOrder ) {
  // --- Compute grid quantities ---
  Grid->compute_mass( uPF );
  Grid->compute_center_of_mass( uPF );

  if ( basis == poly_basis::legendre ) {
    func  = legendre;
    dfunc = d_legendre;
  } else if ( basis == poly_basis::taylor ) {
    func  = taylor;
    dfunc = d_taylor;
  } else {
    THROW_ATHELAS_ERROR( " ! Bad behavior in ModalBasis constructor !" );
  }

  initialize_basis( uPF, Grid );
}

/* --- taylor Methods --- */

/**
 * Return taylor polynomial of given order
 *
 * Parameters:
 * -----------
 * eta   : coordinate
 * eta_c : center of mass
 **/
auto ModalBasis::taylor( const int order, const Real eta, const Real eta_c )
    -> Real {

  if ( order < 0 ) {
    THROW_ATHELAS_ERROR(
        "! polynomial basis :: Please enter a valid polynomial order." );
  }

  // Handle constant and linear terms separately -- no need to exponentiate.
  if ( order == 0 ) {
    return 1.0;
  }
  if ( order == 1 ) {
    return eta - eta_c;
  }
  if ( order > 1 ) {
    return std::pow( eta - eta_c, order );
  }
  return 0.0; // should not be reached.
}

/**
 * Return derivative of taylor polynomial of given order
 *
 * Parameters:
 * -----------
 * eta : coordinate
 * eta_c: center of mass
 **/
auto ModalBasis::d_taylor( const int order, const Real eta, const Real eta_c )
    -> Real {

  if ( order < 0 ) {
    THROW_ATHELAS_ERROR(
        " ! polynomial basis :: Please enter a valid polynomial order." );
  }

  // Handle first few terms separately -- no need to call std::pow
  if ( order == 0 ) {
    return 0.0;
  }
  if ( order == 1 ) {
    return 1.0;
  }
  if ( order == 2 ) {
    return 2 * ( eta - eta_c );
  }
  if ( order > 2 ) {
    return (order)*std::pow( eta - eta_c, order - 1 );
  }
  return 0.0; // should not be reached.
}

/* --- legendre Methods --- */
/* TODO: Make sure that x_c offset for legendre works with COM != 0 */

auto ModalBasis::legendre( const int n, const Real x, const Real x_c ) -> Real {
  return legendre( n, x - x_c );
}

auto ModalBasis::d_legendre( const int n, Real x, const Real x_c ) -> Real {
  return d_legendre( n, x - x_c );
}

// legendre polynomials
auto ModalBasis::legendre( const int n, const Real x ) -> Real {
  return ( n == 0 )   ? 1.0
         : ( n == 1 ) ? x
                      : ( ( ( 2 * n ) - 1 ) * x * legendre( n - 1, x ) -
                          ( n - 1 ) * legendre( n - 2, x ) ) /
                            n;
}

// Derivative of legendre polynomials
auto ModalBasis::d_legendre( const int order, const Real x ) -> Real {

  Real dPn = 0.0; // P_n

  for ( int i = 0; i < order; i++ ) {
    dPn = ( i + 1 ) * legendre( i, x ) + x * dPn;
  }

  return dPn;
}

auto ModalBasis::d_legendre_n( const int poly_order, const int deriv_order,
                               const Real x ) -> Real {
  if ( deriv_order == 0 ) {
    return legendre( poly_order, x );
  }
  if ( poly_order < deriv_order ) {
    return 0.0;
  }
  if ( deriv_order == 1 ) {
    return d_legendre( poly_order, x );
  }
  return ( poly_order * d_legendre_n( poly_order - 1, deriv_order - 1, x ) ) +
         ( x * d_legendre_n( poly_order - 1, deriv_order, x ) );
}

/* TODO: the following 2 inner product functions need to be cleaned */

/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < Psi_m, Phi_n >
 * <f,g> = \sum_q \rho_q f_Q g_q j^0 w_q
 * TODO: Make inner_product functions cleaner????
 **/
auto ModalBasis::inner_product( const int m, const int n, const int iX,
                                const Real eta_c, const View3D<Real> uPF,
                                GridStructure* Grid ) const -> Real {
  Real result = 0.0;
  for ( int iN = 0; iN < nNodes; iN++ ) {
    const Real eta_q = Grid->get_nodes( iN );
    const Real X     = Grid->node_coordinate( iX, iN );
    result += func( n, eta_q, eta_c ) * Phi( iX, iN + 1, m ) *
              Grid->get_weights( iN ) * uPF( 0, iX, iN ) *
              Grid->get_widths( iX ) * Grid->get_sqrt_gm( X );
  }

  return result;
}

/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < Phi_m, Phi_n >
 * <f,g> = \sum_q \rho_q f_q g_q j^0 w_q
 **/
auto ModalBasis::inner_product( const int n, const int iX, const Real /*eta_c*/,
                                const View3D<Real> uPF,
                                GridStructure* Grid ) const -> Real {
  Real result = 0.0;
  for ( int iN = 0; iN < nNodes; iN++ ) {
    const Real X = Grid->node_coordinate( iX, iN );
    result += Phi( iX, iN + 1, n ) * Phi( iX, iN + 1, n ) *
              Grid->get_weights( iN ) * uPF( 0, iX, iN ) *
              Grid->get_widths( iX ) * Grid->get_sqrt_gm( X );
  }

  return result;
}

// Gram-Schmidt orthogonalization of basis
auto ModalBasis::ortho( const int order, const int iX, const int i_eta,
                        const Real eta, const Real eta_c,
                        const View3D<Real> uPF, GridStructure* Grid,
                        bool const derivative_option ) -> Real {

  Real result = 0.0;

  // TODO(astrobarker): Can this be cleaned up?
  if ( not derivative_option ) {
    result = func( order, eta, eta_c );
  } else {
    result = dfunc( order, eta, eta_c );
  }

  // if ( order == 0 ) return result;

  Real phi_n = 0.0;
  for ( int k = 0; k < order; k++ ) {
    const Real numerator =
        inner_product( order - k - 1, order, iX, eta_c, uPF, Grid );
    const Real denominator =
        inner_product( order - k - 1, iX, eta_c, uPF, Grid );
    // ? Can this be cleaned up?
    if ( !derivative_option ) {
      phi_n = Phi( iX, i_eta, order - k - 1 );
    }
    if ( derivative_option ) {
      phi_n = dPhi( iX, i_eta, order - k - 1 );
    }
    result -= ( numerator / denominator ) * phi_n;
  }

  return result;
}

/**
 * Pre-compute the orthogonal basis terms. Phi(iX,k,eta) will store
 * the expansion terms for each order k, stored at various points eta.
 * We store: (-0.5, {GL nodes}, 0.5) for a total of nNodes+2
 * TODO: Incorporate COM centering?
 **/
void ModalBasis::initialize_basis( const Kokkos::View<Real***> uPF,
                                   GridStructure* Grid ) {
  const int n_eta = ( 3 * nNodes ) + 2;
  const int ilo   = Grid->get_ilo( );
  const int ihi   = Grid->get_ihi( );

  Real eta = 0.5;
  for ( int iX = ilo; iX <= ihi; iX++ ) {
    for ( int k = 0; k < order; k++ ) {
      for ( int i_eta = 0; i_eta < n_eta; i_eta++ ) {
        // face values
        if ( i_eta == 0 ) {
          eta = -0.5;
        } else if ( i_eta == nNodes + 1 ) {
          eta = +0.5;
        } else if ( i_eta > 0 && i_eta < nNodes + 1 ) // GL nodes
        {
          eta = Grid->get_nodes( i_eta - 1 );
        } else if ( i_eta > nNodes + 1 &&
                    i_eta < 2 * nNodes + 2 ) // GL nodes left neighbor
        {
          eta = Grid->get_nodes( i_eta - nNodes - 2 ) + 1.0;
        } else {
          eta = Grid->get_nodes( i_eta - ( 2 * nNodes ) - 2 ) - 1.0;
        }

        Phi( iX, i_eta, k ) = ortho( k, iX, i_eta, eta, 0.0, uPF, Grid, false );
        dPhi( iX, i_eta, k ) = ortho( k, iX, i_eta, eta, 0.0, uPF, Grid, true );
      }
    }
  }
  check_orthogonality( uPF, Grid );
  compute_mass_matrix( uPF, Grid );

  // === Fill Guard cells ===

  // ? Using identical basis in guard cells as boundaries ?
  for ( int iX = 0; iX < ilo; iX++ ) {
    for ( int i_eta = 0; i_eta < n_eta; i_eta++ ) {
      for ( int k = 0; k < order; k++ ) {
        Phi( ilo - 1 - iX, i_eta, k ) = Phi( ilo + iX, i_eta, k );
        Phi( ihi + 1 + iX, i_eta, k ) = Phi( ihi - iX, i_eta, k );

        dPhi( ilo - 1 - iX, i_eta, k ) = dPhi( ilo + iX, i_eta, k );
        dPhi( ihi + 1 + iX, i_eta, k ) = dPhi( ihi - iX, i_eta, k );
      }
    }
  }

  for ( int iX = 0; iX < ilo; iX++ ) {
    for ( int k = 0; k < order; k++ ) {
      MassMatrix( ilo - 1 - iX, k ) = MassMatrix( ilo + iX, k );
      MassMatrix( ihi + 1 + iX, k ) = MassMatrix( ihi - iX, k );
    }
  }
}

/**
 * The following checks orthogonality of basis functions on each cell.
 * Returns error if orthogonality is not met.
 **/
void ModalBasis::check_orthogonality( const Kokkos::View<Real***> uPF,
                                      GridStructure* Grid ) const {

  const int ilo = Grid->get_ilo( );
  const int ihi = Grid->get_ihi( );

  for ( int iX = ilo; iX <= ihi; iX++ ) {
    for ( int k1 = 0; k1 < order; k1++ ) {
      for ( int k2 = 0; k2 < order; k2++ ) {
        Real result = 0.0;
        for ( int i_eta = 1; i_eta <= nNodes; i_eta++ ) // loop over quadratures
        {
          const Real X = Grid->node_coordinate( iX, i_eta - 1 );
          // Not using an inner_product function because their API is odd..
          result += Phi( iX, i_eta, k1 ) * Phi( iX, i_eta, k2 ) *
                    uPF( 0, iX, i_eta - 1 ) * Grid->get_weights( i_eta - 1 ) *
                    Grid->get_widths( iX ) * Grid->get_sqrt_gm( X );
        }

        if ( k1 == k2 && result == 0.0 ) {
          THROW_ATHELAS_ERROR(
              " ! Basis not orthogonal: Diagonal term equal to zero.\n" );
        }
        if ( k1 != k2 && std::abs( result ) > 1e-10 ) {
          std::println( "{} {} {:.3e}", k1, k2, result );
          THROW_ATHELAS_ERROR(
              " ! Basis not orthogonal: Off diagonal term non-zero.\n" );
        }
      }
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
void ModalBasis::compute_mass_matrix( const View3D<Real> uPF,
                                      GridStructure* Grid ) {
  const int ilo    = Grid->get_ilo( );
  const int ihi    = Grid->get_ihi( );
  const int nNodes = Grid->get_n_nodes( );

  for ( int iX = ilo; iX <= ihi; iX++ ) {
    for ( int k = 0; k < order; k++ ) {
      Real result = 0.0;
      for ( int iN = 0; iN < nNodes; iN++ ) {
        const Real X = Grid->node_coordinate( iX, iN );
        result += Phi( iX, iN + 1, k ) * Phi( iX, iN + 1, k ) *
                  Grid->get_weights( iN ) * Grid->get_widths( iX ) *
                  Grid->get_sqrt_gm( X ) * uPF( 0, iX, iN );
      }
      MassMatrix( iX, k ) = result;
    }
  }
}

/**
 * Evaluate (modal) basis on element iX for quantity iCF.
 **/
auto ModalBasis::basis_eval( View3D<Real> U, const int iX, const int iCF,
                             const int i_eta ) const -> Real {
  Real result = 0.0;
  for ( int k = 0; k < order; k++ ) {
    result += Phi( iX, i_eta, k ) * U( iCF, iX, k );
  }
  return result;
}

// Same as above, for a 2D vector U_k on a given cell and quantity
// e.g., U(:, iX, :)
auto ModalBasis::basis_eval( View2D<Real> U, const int iX, const int iCF,
                             const int i_eta ) const -> Real {
  Real result = 0.0;
  for ( int k = 0; k < order; k++ ) {
    result += Phi( iX, i_eta, k ) * U( iCF, k );
  }
  return result;
}

// Same as above, for a 1D vector U_k on a given cell and quantity
// e.g., U(iCF, iX, :)
auto ModalBasis::basis_eval( View1D<Real> U, const int iX,
                             const int i_eta ) const -> Real {
  Real result = 0.0;
  for ( int k = 0; k < order; k++ ) {
    result += Phi( iX, i_eta, k ) * U( k );
  }
  return result;
}

// Accessor for Phi
auto ModalBasis::get_phi( const int iX, const int i_eta, const int k ) const
    -> Real {
  return Phi( iX, i_eta, k );
}

// Accessor for dPhi
auto ModalBasis::get_d_phi( const int iX, const int i_eta, const int k ) const
    -> Real {
  return dPhi( iX, i_eta, k );
}

// Accessor for mass matrix
auto ModalBasis::get_mass_matrix( const int iX, const int k ) const -> Real {
  return MassMatrix( iX, k );
}

// Accessor for Order
auto ModalBasis::get_order( ) const noexcept -> int { return order; }
