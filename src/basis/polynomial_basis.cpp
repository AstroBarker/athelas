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
 *
 * TODO(astrobarker): kokkos
 * TODO(astrobarker): need center of mass for some probs?
 * TODO(astrobarker): derivative matrix
 */

#include <cmath>
#include <cstdlib>
#include <print>

#include "abstractions.hpp"
#include "error.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"

/**
 * Constructor creates necessary matrices and bases, etc.
 * This has to be called after the problem is initialized.
 **/
ModalBasis::ModalBasis(poly_basis::poly_basis basis, const View3D<double> uPF,
                       GridStructure* grid, const int pOrder, const int nN,
                       const int nElements, const int nGuard,
                       const bool density_weight)
    : nX_(nElements), order_(pOrder), nNodes_(nN),
      mSize_((nN) * (nN + 2) * (nElements + 2 * nGuard)),
      density_weight_(density_weight),
      mass_matrix_("MassMatrix", nElements + 2 * nGuard, pOrder),
      phi_("phi_", nElements + 2 * nGuard, 3 * nN + 2, pOrder),
      dphi_("dphi_", nElements + 2 * nGuard, 3 * nN + 2, pOrder) {
  // --- Compute grid quantities ---
  grid->compute_mass(uPF);
  grid->compute_mass_r(uPF); // Weird place for this to be but works
  grid->compute_center_of_mass(uPF);

  if (basis == poly_basis::legendre) {
    func_  = legendre;
    dfunc_ = d_legendre;
  } else if (basis == poly_basis::taylor) {
    func_  = taylor;
    dfunc_ = d_taylor;
  } else {
    THROW_ATHELAS_ERROR(" ! Bad behavior in ModalBasis constructor !");
  }

  initialize_basis(uPF, grid);
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
auto ModalBasis::taylor(const int order, const double eta, const double eta_c)
    -> double {

  if (order < 0) {
    THROW_ATHELAS_ERROR(
        "! polynomial basis :: Please enter a valid polynomial order.");
  }

  // Handle constant and linear terms separately -- no need to exponentiate.
  if (order == 0) {
    return 1.0;
  }
  if (order == 1) {
    return eta - eta_c;
  }
  if (order > 1) {
    return std::pow(eta - eta_c, order);
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
auto ModalBasis::d_taylor(const int order, const double eta, const double eta_c)
    -> double {

  if (order < 0) {
    THROW_ATHELAS_ERROR(
        " ! polynomial basis :: Please enter a valid polynomial order.");
  }

  // Handle first few terms separately -- no need to call std::pow
  if (order == 0) {
    return 0.0;
  }
  if (order == 1) {
    return 1.0;
  }
  if (order == 2) {
    return 2 * (eta - eta_c);
  }
  if (order > 2) {
    return (order)*std::pow(eta - eta_c, order - 1);
  }
  return 0.0; // should not be reached.
}

/* --- legendre Methods --- */
/* TODO: Make sure that x_c offset for legendre works with COM != 0 */

auto ModalBasis::legendre(const int n, const double x, const double x_c)
    -> double {
  return legendre(n, x - x_c);
}

auto ModalBasis::d_legendre(const int n, double x, const double x_c) -> double {
  return d_legendre(n, x - x_c);
}

// legendre polynomials
auto ModalBasis::legendre(const int n, const double x) -> double {
  return (n == 0)   ? 1.0
         : (n == 1) ? x
                    : (((2 * n) - 1) * x * legendre(n - 1, x) -
                       (n - 1) * legendre(n - 2, x)) /
                          n;
}

// Derivative of legendre polynomials
auto ModalBasis::d_legendre(const int order, const double x) -> double {

  double dPn = 0.0; // P_n

  for (int i = 0; i < order; i++) {
    dPn = (i + 1) * legendre(i, x) + x * dPn;
  }

  return dPn;
}

auto ModalBasis::d_legendre_n(const int poly_order, const int deriv_order,
                              const double x) -> double {
  if (deriv_order == 0) {
    return legendre(poly_order, x);
  }
  if (poly_order < deriv_order) {
    return 0.0;
  }
  if (deriv_order == 1) {
    return d_legendre(poly_order, x);
  }
  return (poly_order * d_legendre_n(poly_order - 1, deriv_order - 1, x)) +
         (x * d_legendre_n(poly_order - 1, deriv_order, x));
}

/* TODO: the following 2 inner product functions need to be cleaned */

/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < Psi_m, phi__n >
 * <f,g> = \sum_q \rho_q f_Q g_q j^0 w_q
 * TODO: Make inner_product functions cleaner????
 **/
auto ModalBasis::inner_product(const int m, const int n, const int iX,
                               const double eta_c, const View3D<double> uPF,
                               GridStructure* grid) const -> double {
  double result = 0.0;
  for (int iN = 0; iN < nNodes_; iN++) {
    // include rho in integrand if necessary
    const double rho   = density_weight_ ? uPF(0, iX, iN) : 1.0;
    const double eta_q = grid->get_nodes(iN);
    const double X     = grid->node_coordinate(iX, iN);
    result += func_(n, eta_q, eta_c) * phi_(iX, iN + 1, m) *
              grid->get_weights(iN) * rho * grid->get_widths(iX) *
              grid->get_sqrt_gm(X);
  }

  return result;
}

/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < phi__m, phi__n >
 * <f,g> = \sum_q \rho_q f_q g_q j^0 w_q
 **/
auto ModalBasis::inner_product(const int n, const int iX,
                               const double /*eta_c*/, const View3D<double> uPF,
                               GridStructure* grid) const -> double {
  double result = 0.0;
  for (int iN = 0; iN < nNodes_; iN++) {
    // include rho in integrand if necessary
    const double rho = density_weight_ ? uPF(0, iX, iN) : 1.0;
    const double X   = grid->node_coordinate(iX, iN);
    result += phi_(iX, iN + 1, n) * phi_(iX, iN + 1, n) *
              grid->get_weights(iN) * rho * grid->get_widths(iX) *
              grid->get_sqrt_gm(X);
  }

  return result;
}

// Gram-Schmidt orthogonalization of basis
auto ModalBasis::ortho(const int order, const int iX, const int i_eta,
                       const double eta, const double eta_c,
                       const View3D<double> uPF, GridStructure* grid,
                       bool const derivative_option) -> double {

  double result = 0.0;

  // TODO(astrobarker): Can this be cleaned up?
  if (not derivative_option) {
    result = func_(order, eta, eta_c);
  } else {
    result = dfunc_(order, eta, eta_c);
  }

  // if ( order == 0 ) return result;

  double phi_n = 0.0;
  for (int k = 0; k < order; k++) {
    const double numerator =
        inner_product(order - k - 1, order, iX, eta_c, uPF, grid);
    const double denominator =
        inner_product(order - k - 1, iX, eta_c, uPF, grid);
    // ? Can this be cleaned up?
    if (!derivative_option) {
      phi_n = phi_(iX, i_eta, order - k - 1);
    }
    if (derivative_option) {
      phi_n = dphi_(iX, i_eta, order - k - 1);
    }
    result -= (numerator / denominator) * phi_n;
  }

  return result;
}

/**
 * Pre-compute the orthogonal basis terms. phi_(iX,k,eta) will store
 * the expansion terms for each order k, stored at various points eta.
 * We store: (-0.5, {GL nodes}, 0.5) for a total of nNodes+2
 * TODO: Incorporate COM centering?
 **/
void ModalBasis::initialize_basis(const Kokkos::View<double***> uPF,
                                  GridStructure* grid) {
  const int n_eta = (3 * nNodes_) + 2;
  const int ilo   = grid->get_ilo();
  const int ihi   = grid->get_ihi();

  double eta = 0.5;
  for (int iX = ilo; iX <= ihi; iX++) {
    for (int k = 0; k < order_; k++) {
      for (int i_eta = 0; i_eta < n_eta; i_eta++) {
        // face values
        if (i_eta == 0) {
          eta = -0.5;
        } else if (i_eta == nNodes_ + 1) {
          eta = +0.5;
        } else if (i_eta > 0 && i_eta < nNodes_ + 1) // GL nodes
        {
          eta = grid->get_nodes(i_eta - 1);
        } else if (i_eta > nNodes_ + 1 &&
                   i_eta < 2 * nNodes_ + 2) // GL nodes left neighbor
        {
          eta = grid->get_nodes(i_eta - nNodes_ - 2) + 1.0;
        } else {
          eta = grid->get_nodes(i_eta - (2 * nNodes_) - 2) - 1.0;
        }

        phi_(iX, i_eta, k)  = ortho(k, iX, i_eta, eta, 0.0, uPF, grid, false);
        dphi_(iX, i_eta, k) = ortho(k, iX, i_eta, eta, 0.0, uPF, grid, true);
      }
    }
  }
  check_orthogonality(uPF, grid);
  compute_mass_matrix(uPF, grid);

  // === Fill Guard cells ===

  // ? Using identical basis in guard cells as boundaries ?
  for (int iX = 0; iX < ilo; iX++) {
    for (int i_eta = 0; i_eta < n_eta; i_eta++) {
      for (int k = 0; k < order_; k++) {
        phi_(ilo - 1 - iX, i_eta, k) = phi_(ilo + iX, i_eta, k);
        phi_(ihi + 1 + iX, i_eta, k) = phi_(ihi - iX, i_eta, k);

        dphi_(ilo - 1 - iX, i_eta, k) = dphi_(ilo + iX, i_eta, k);
        dphi_(ihi + 1 + iX, i_eta, k) = dphi_(ihi - iX, i_eta, k);
      }
    }
  }

  for (int iX = 0; iX < ilo; iX++) {
    for (int k = 0; k < order_; k++) {
      mass_matrix_(ilo - 1 - iX, k) = mass_matrix_(ilo + iX, k);
      mass_matrix_(ihi + 1 + iX, k) = mass_matrix_(ihi - iX, k);
    }
  }
}

/**
 * The following checks orthogonality of basis functions on each cell.
 * Returns error if orthogonality is not met.
 **/
void ModalBasis::check_orthogonality(const Kokkos::View<double***> uPF,
                                     GridStructure* grid) const {

  const int ilo = grid->get_ilo();
  const int ihi = grid->get_ihi();

  for (int iX = ilo; iX <= ihi; iX++) {
    for (int k1 = 0; k1 < order_; k1++) {
      for (int k2 = 0; k2 < order_; k2++) {
        double result = 0.0;
        for (int i_eta = 1; i_eta <= nNodes_; i_eta++) // loop over quadratures
        {
          const double rho = density_weight_ ? uPF(0, iX, i_eta - 1) : 1.0;
          const double X   = grid->node_coordinate(iX, i_eta - 1);
          // Not using an inner_product function because their API is odd..
          result += phi_(iX, i_eta, k1) * phi_(iX, i_eta, k2) * rho *
                    grid->get_weights(i_eta - 1) * grid->get_widths(iX) *
                    grid->get_sqrt_gm(X);
        }

        if (k1 == k2 && result == 0.0) {
          THROW_ATHELAS_ERROR(
              " ! Basis not orthogonal: Diagonal term equal to zero.\n");
        }
        if (k1 != k2 && std::abs(result) > 1e-10) {
          std::println("{} {} {:.3e}", k1, k2, result);
          THROW_ATHELAS_ERROR(
              " ! Basis not orthogonal: Off diagonal term non-zero.\n");
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
void ModalBasis::compute_mass_matrix(const View3D<double> uPF,
                                     GridStructure* grid) {
  const int ilo     = grid->get_ilo();
  const int ihi     = grid->get_ihi();
  const int nNodes_ = grid->get_n_nodes();

  for (int iX = ilo; iX <= ihi; iX++) {
    for (int k = 0; k < order_; k++) {
      double result = 0.0;
      for (int iN = 0; iN < nNodes_; iN++) {
        // include rho in integrand if necessary
        const double rho = density_weight_ ? uPF(0, iX, iN) : 1.0;
        const double X   = grid->node_coordinate(iX, iN);
        result += phi_(iX, iN + 1, k) * phi_(iX, iN + 1, k) *
                  grid->get_weights(iN) * grid->get_widths(iX) *
                  grid->get_sqrt_gm(X) * rho;
      }
      mass_matrix_(iX, k) = result;
    }
  }
}

/**
 * Evaluate (modal) basis on element iX for quantity iCF.
 **/
auto ModalBasis::basis_eval(View3D<double> U, const int iX, const int iCF,
                            const int i_eta) const -> double {
  double result = 0.0;
  for (int k = 0; k < order_; k++) {
    result += phi_(iX, i_eta, k) * U(iCF, iX, k);
  }
  return result;
}

// Same as above, for a 2D vector U_k on a given cell and quantity
// e.g., U(:, iX, :)
auto ModalBasis::basis_eval(View2D<double> U, const int iX, const int iCF,
                            const int i_eta) const -> double {
  double result = 0.0;
  for (int k = 0; k < order_; k++) {
    result += phi_(iX, i_eta, k) * U(iCF, k);
  }
  return result;
}

// Same as above, for a 1D vector U_k on a given cell and quantity
// e.g., U(iCF, iX, :)
auto ModalBasis::basis_eval(View1D<double> U, const int iX,
                            const int i_eta) const -> double {
  double result = 0.0;
  for (int k = 0; k < order_; k++) {
    result += phi_(iX, i_eta, k) * U(k);
  }
  return result;
}

// Accessor for phi_
auto ModalBasis::get_phi(const int iX, const int i_eta, const int k) const
    -> double {
  return phi_(iX, i_eta, k);
}

// Accessor for dphi_
auto ModalBasis::get_d_phi(const int iX, const int i_eta, const int k) const
    -> double {
  return dphi_(iX, i_eta, k);
}

// Accessor for mass matrix
auto ModalBasis::get_mass_matrix(const int iX, const int k) const -> double {
  return mass_matrix_(iX, k);
}

// Accessor for Order
auto ModalBasis::get_order() const noexcept -> int { return order_; }
