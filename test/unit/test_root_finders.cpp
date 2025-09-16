#include <print>

#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"

#include "solvers/root_finders.hpp"

using root_finders::AbsoluteError, root_finders::RelativeError,
    root_finders::HybridError, root_finders::RootFinder,
    root_finders::NewtonAlgorithm, root_finders::AANewtonAlgorithm,
    root_finders::FixedPointAlgorithm, root_finders::AAFixedPointAlgorithm;

constexpr double sqrt2 = std::numbers::sqrt2;
constexpr double cos_fixed_point = 0.7390851332151607;

constexpr double tol = 1e-10;

TEST_CASE("Newton algorithm: x^2 - 2", "[newton]") {
  auto f = [](double x) { return x * x - 2.0; };
  auto df = [](double x) { return 2.0 * x; };

  RootFinder<double, NewtonAlgorithm<double>> solver;
  solver.set_tolerance(1e-14, 1e-14).set_max_iterations(100);

  double root = solver.solve(f, df, 2.0);
  REQUIRE(soft_equal(root, sqrt2, tol));
}

TEST_CASE("Anderson accelerated Newton algorithm: x^2 - 2", "[aa_newton]") {
  auto f = [](double x) { return x * x - 2.0; };
  auto df = [](double x) { return 2.0 * x; };

  RootFinder<double, AANewtonAlgorithm<double>> solver;
  solver.set_tolerance(1e-14, 1e-14).set_max_iterations(100);

  double root = solver.solve(f, df, 2.0);
  REQUIRE(soft_equal(root, sqrt2, tol));
}

TEST_CASE("Fixed point iteration for x = cos(x)", "[fixed_point]") {
  auto g = [](double x) { return std::cos(x); };

  RootFinder<double, FixedPointAlgorithm<double>> solver;
  solver.set_tolerance(1e-12, 1e-12).set_max_iterations(100);

  double root = solver.solve(g, 2.0);
  REQUIRE(soft_equal(root, cos_fixed_point, tol));
}

TEST_CASE("Anderson accelerated fixed point iteration for x = cos(x)",
          "[aa_fixed_point]") {
  auto g = [](double x) { return std::cos(x); };

  RootFinder<double, AAFixedPointAlgorithm<double>> solver;
  solver.set_tolerance(1e-12, 1e-12).set_max_iterations(100);

  double root = solver.solve(g, 2.0);
  REQUIRE(soft_equal(root, cos_fixed_point, tol));
}
