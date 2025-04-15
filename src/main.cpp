#include <print>
#include <string>

#include "Kokkos_Core.hpp"

#include "bc/boundary_conditions.hpp"
#include "driver.hpp"
#include "io/io.hpp"
#include "main.hpp"
#include "utils/error.hpp"

namespace {
auto parse_input_file(int argc, char* argv[]) -> std::string {
    std::string input_path;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            input_path = argv[++i];
        } else {
            std::println(stderr, "Unknown or malformed option: {}", arg);
            std::println(stderr, "Usage: ./athelas -i input_file.toml");
            std::exit(1);
        }
    }

    if (input_path.empty()) {
        std::println(stderr, "Missing required input file.");
        std::println(stderr, "Usage: ./athelas -i input_file.toml");
        std::exit(1);
    }

    return input_path;
}
} // namespace

using bc::apply_bc;

auto main( int argc, char** argv ) -> int {
  const std::string input_file = parse_input_file(argc, argv);

  auto sig1 = signal( SIGSEGV, segfault_handler );
  auto sig2 = signal( SIGABRT, segfault_handler );

  // create span of args
  auto args = std::span( argv, static_cast<size_t>( argc ) );

  Kokkos::initialize(argc, argv);
  {
    // pin
    const auto pin = std::make_unique<ProblemIn>(input_file);

    // --- Create Driver ---
    Driver driver(pin.get());

    // --- Timer ---
    Kokkos::Timer timer_total;

    // --- execute driver ---
    driver.execute();

    // --- Finalize timer ---
    Real const time = timer_total.seconds( );
    std::println( " ~ Done! Elapsed time: {} seconds.", time );

  }
  Kokkos::finalize();

  return AthelasExitCodes::SUCCESS;
} // main

