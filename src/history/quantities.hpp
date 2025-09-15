/**
 * @file quantities.hpp
 * --------------
 *
 * @brief Select quantity calcuations for history
 *
 * TODO(astrobarker): track boundary fluxes
 * TODO(astrobarker): Loop is 4 pi
 */

#include "geometry/grid.hpp"
#include "polynomial_basis.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace analysis {

// Perhaps the below will be more optimal by calculating
// with cell mass
KOKKOS_INLINE_FUNCTION
auto total_fluid_energy(const State &state, const GridStructure &grid,
                        const ModalBasis *fluid_basis,
                        const ModalBasis * /*rad_basis*/) -> double {
  const auto &ilo = grid.get_ilo();
  const auto &ihi = grid.get_ihi();
  const auto &nNodes = grid.get_n_nodes();

  const auto u = state.u_cf();

  double output = 0.0;
  Kokkos::parallel_reduce(
      "History :: TotalEnergyFluid", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(const int &i, double &lsum) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(i, iN);
          local_sum += fluid_basis->basis_eval(u, i, 2, iN + 1) /
                       fluid_basis->basis_eval(u, i, 0, iN + 1) *
                       grid.get_sqrt_gm(X) * grid.get_weights(iN);
        }
        lsum += local_sum * grid.get_widths(i);
      },
      output);

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

KOKKOS_INLINE_FUNCTION
auto total_fluid_momentum(const State &state, const GridStructure &grid,
                          const ModalBasis *fluid_basis,
                          const ModalBasis * /*rad_basis*/) -> double {
  const auto &ilo = grid.get_ilo();
  const auto &ihi = grid.get_ihi();
  const auto &nNodes = grid.get_n_nodes();

  const auto u = state.u_cf();

  double output = 0.0;
  Kokkos::parallel_reduce(
      "History :: TotalMomentumFluid", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(const int &i, double &lsum) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(i, iN);
          local_sum += fluid_basis->basis_eval(u, i, 1, iN + 1) /
                       fluid_basis->basis_eval(u, i, 0, iN + 1) *
                       grid.get_sqrt_gm(X) * grid.get_weights(iN);
        }
        lsum += local_sum * grid.get_widths(i);
      },
      output);

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

KOKKOS_INLINE_FUNCTION
auto total_internal_energy(const State &state, const GridStructure &grid,
                           const ModalBasis *fluid_basis,
                           const ModalBasis * /*rad_basis*/) -> double {
  const auto &ilo = grid.get_ilo();
  const auto &ihi = grid.get_ihi();
  const auto &nNodes = grid.get_n_nodes();

  const auto u = state.u_cf();

  double output = 0.0;
  Kokkos::parallel_reduce(
      "History :: TotalInternalEnergy", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(const int &i, double &lsum) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(i, iN);
          const double vel = fluid_basis->basis_eval(u, i, 1, iN + 1);
          local_sum +=
              (fluid_basis->basis_eval(u, i, 2, iN + 1) - 0.5 * vel * vel) /
              fluid_basis->basis_eval(u, i, 0, iN + 1) * grid.get_sqrt_gm(X) *
              grid.get_weights(iN);
        }
        lsum += local_sum * grid.get_widths(i);
      },
      output);

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

KOKKOS_INLINE_FUNCTION
auto total_gravitational_energy(const State &state, const GridStructure &grid,
                                const ModalBasis *fluid_basis,
                                const ModalBasis * /*rad_basis*/) -> double {
  const auto &ilo = grid.get_ilo();
  const auto &ihi = grid.get_ihi();
  const auto &nNodes = grid.get_n_nodes();

  const auto u = state.u_cf();

  double output = 0.0;
  Kokkos::parallel_reduce(
      "History :: TotalGravitationalEnergy",
      Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(const int &i, double &lsum) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(i, iN);
          local_sum += (grid.enclosed_mass(i, iN) /
                        (X / fluid_basis->basis_eval(u, i, 0, iN + 1))) *
                       grid.get_sqrt_gm(X) * grid.get_weights(iN);
        }
        lsum += local_sum * grid.get_widths(i);
      },
      output);

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return -constants::G_GRAV * output;
}

KOKKOS_INLINE_FUNCTION
auto total_kinetic_energy(const State &state, const GridStructure &grid,
                          const ModalBasis *fluid_basis,
                          const ModalBasis * /*rad_basis*/) -> double {
  const auto &ilo = grid.get_ilo();
  const auto &ihi = grid.get_ihi();
  const auto &nNodes = grid.get_n_nodes();

  const auto u = state.u_cf();

  double output = 0.0;
  Kokkos::parallel_reduce(
      "History :: TotalKineticEnergy", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(const int &i, double &lsum) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(i, iN);
          const double vel = fluid_basis->basis_eval(u, i, 1, iN + 1);
          local_sum += (0.5 * vel * vel) /
                       fluid_basis->basis_eval(u, i, 0, iN + 1) *
                       grid.get_sqrt_gm(X) * grid.get_weights(iN);
        }
        lsum += local_sum * grid.get_widths(i);
      },
      output);

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

// This total_energy is only radiation
KOKKOS_INLINE_FUNCTION
auto total_rad_energy(const State &state, const GridStructure &grid,
                      const ModalBasis * /*fluid_basis*/,
                      const ModalBasis *rad_basis) -> double {
  const auto &ilo = grid.get_ilo();
  const auto &ihi = grid.get_ihi();
  const auto &nNodes = grid.get_n_nodes();

  const auto u = state.u_cf();

  double output = 0.0;
  Kokkos::parallel_reduce(
      "History :: TotalEnergyRad", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(const int &i, double &lsum) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(i, iN);
          local_sum += rad_basis->basis_eval(u, i, 3, iN + 1) *
                       grid.get_sqrt_gm(X) * grid.get_weights(iN);
        }
        lsum += local_sum * grid.get_widths(i);
      },
      output);

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

// TODO(astrobarker): confirm
KOKKOS_INLINE_FUNCTION
auto total_rad_momentum(const State &state, const GridStructure &grid,
                        const ModalBasis * /*fluid_basis*/,
                        const ModalBasis *rad_basis) -> double {
  const auto &ilo = grid.get_ilo();
  const auto &ihi = grid.get_ihi();
  const auto &nNodes = grid.get_n_nodes();

  const auto u = state.u_cf();

  double output = 0.0;
  Kokkos::parallel_reduce(
      "History :: TotalRadMomentum", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(const int &i, double &lsum) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(i, iN);
          local_sum += rad_basis->basis_eval(u, i, 4, iN + 1) *
                       grid.get_sqrt_gm(X) * grid.get_weights(iN);
        }
        lsum += local_sum * grid.get_widths(i);
      },
      output);

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

// This total_energy is matter and radiation
KOKKOS_INLINE_FUNCTION
auto total_energy(const State &state, const GridStructure &grid,
                  const ModalBasis *fluid_basis, const ModalBasis *rad_basis)
    -> double {
  const auto &ilo = grid.get_ilo();
  const auto &ihi = grid.get_ihi();
  const auto &nNodes = grid.get_n_nodes();

  const auto u = state.u_cf();

  double output = 0.0;
  Kokkos::parallel_reduce(
      "History :: TotalEnergy", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(const int &i, double &lsum) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(i, iN);
          local_sum += ((fluid_basis->basis_eval(u, i, 2, iN + 1) /
                         fluid_basis->basis_eval(u, i, 0, iN + 1)) +
                        rad_basis->basis_eval(u, i, 3, iN + 1)) *
                       grid.get_sqrt_gm(X) * grid.get_weights(iN);
        }
        lsum += local_sum * grid.get_widths(i);
      },
      output);

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

// This total_energy is matter and radiation
KOKKOS_INLINE_FUNCTION
auto total_momentum(const State &state, const GridStructure &grid,
                    const ModalBasis *fluid_basis, const ModalBasis *rad_basis)
    -> double {
  const auto &ilo = grid.get_ilo();
  const auto &ihi = grid.get_ihi();
  const auto &nNodes = grid.get_n_nodes();

  const auto u = state.u_cf();

  double output = 0.0;
  Kokkos::parallel_reduce(
      "History :: TotalMomentum", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(const int &i, double &lsum) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(i, iN);
          local_sum += ((fluid_basis->basis_eval(u, i, 1, iN + 1) /
                         fluid_basis->basis_eval(u, i, 0, iN + 1)) +
                        rad_basis->basis_eval(u, i, 4, iN + 1)) *
                       grid.get_sqrt_gm(X) * grid.get_weights(iN);
        }
        lsum += local_sum * grid.get_widths(i);
      },
      output);

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

KOKKOS_INLINE_FUNCTION
auto total_mass(const State &state, const GridStructure &grid,
                const ModalBasis *fluid_basis, const ModalBasis * /*rad_basis*/)
    -> double {
  const auto &ilo = grid.get_ilo();
  const auto &ihi = grid.get_ihi();
  const auto &nNodes = grid.get_n_nodes();

  const auto u = state.u_cf();

  double output = 0.0;
  Kokkos::parallel_reduce(
      "History :: TotalMass", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_LAMBDA(const int &i, double &lsum) {
        double local_sum = 0.0;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double X = grid.node_coordinate(i, iN);
          local_sum += (1.0 / fluid_basis->basis_eval(u, i, 0, iN + 1)) *
                       grid.get_sqrt_gm(X) * grid.get_weights(iN);
        }
        lsum += local_sum * grid.get_widths(i);
      },
      output);

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}
} // namespace analysis
