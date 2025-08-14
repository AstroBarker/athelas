#include "hydrostatic_equilibrium.hpp"
#include "Kokkos_Core.hpp"
#include "utils/constants.hpp"

auto HydrostaticEquilibrium::rhs(const double mass_enc, const double p, const double r) const -> double {
  static constexpr double G = constants::G_GRAV;
  const double rho = std::pow(p/k_, n_/(n_ + 1.0));
  return - G * mass_enc * rho / (r * r);
}

void HydrostaticEquilibrium::solve(View3D<double> uAF, const GridStructure* grid) {
  static constexpr int ilo = 1;
  const int ihi = grid->get_ihi();
  const int nNodes = grid->get_n_nodes();
  // Subtely: do we associate rho_c_ with the inner boundary or first nodal point?
  const double vel = 0.0;
  const double energy = 0.0;
  const auto lambda = nullptr;
  const double p_c = pressure_from_conserved(eos_, rho_c_, vel, energy, lambda);
  std::println("rhoc pc {} {}", rho_c_, p_c);

  const double r_c = grid->node_coordinate(ilo,0);
  double m_enc = (constants::FOURPI / 3.0) * (r_c * r_c * r_c) * rho_c_;

  // host mirrors;
  auto h_uAF = Kokkos::create_mirror_view(uAF);

  const int size = grid->get_n_elements() * grid->get_n_nodes();
  View1D<double> d_r("host radius", size);
  View1D<double> d_p("host pressure", size);
  Kokkos::parallel_for("copy grid",
      Kokkos::RangePolicy<>(0, ihi + 1), KOKKOS_LAMBDA(int iX) {
      for (int iN = 0; iN < nNodes; ++iN) {
        const double r = grid->node_coordinate(iX + 1, iN);
        d_r(iX * nNodes + iN) = r;
      }
  });
  auto h_r = Kokkos::create_mirror_view(d_r);
  auto h_p = Kokkos::create_mirror_view(d_p);
  Kokkos::deep_copy(h_r, d_r);
  Kokkos::deep_copy(h_p, d_p);
  h_p(0) = p_c;
  
  for (int i = 0; i < size; ++i) {
    const double r = h_r(i);
    const double p = h_p(i);
    const double rho = std::pow(p/k_, n_/(n_ + 1.0));
    const double dr = h_r(i + 1) - r;
    std::println("p, rho = {} {}", h_p(i), rho);
    const double m = m_enc;// + constants::FOURPI * rho * r * r * dr;

    // RK4
    // MAKE RHS IN TERMS OF P
    // NOTE: Currently holding m constant through the stages!
    const double k1 = dr * rhs(m, p, r);
    const double k2 = dr * rhs(m, p + 0.5  * k1, r + 0.5 * dr);
    const double k3 = dr * rhs(m, p + 0.5  * k2, r + 0.5 * dr);
    const double k4 = dr * rhs(m, p + k3, r + dr);

    m_enc += constants::FOURPI * rho * r * r * dr;
    h_p(i+1) = std::max(p + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), p_threshold_);
  }
}
