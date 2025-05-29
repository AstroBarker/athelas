#include <expected>
#include <print>
#include <string>

#include "Kokkos_Core.hpp"

#include "bc/boundary_conditions.hpp"
#include "driver.hpp"
#include "io/io.hpp"
#include "main.hpp"
#include "utils/error.hpp"

namespace {
auto parse_input_file( std::span<char*> args )
    -> std::expected<std::string, std::string> {
  for ( std::size_t i = 1; i < args.size( ); ++i ) {
    std::string_view arg = args[i];
    if ( arg == "-i" || arg == "--input" ) {
      if ( i + 1 >= args.size( ) ) {
        return std::unexpected( "Missing input file after -i" );
      }
      return std::string( args[i + 1] );
    }
  }
  return std::unexpected( "No input file passed! Use -i <path>" );
}
} // namespace

using bc::apply_bc;

auto main( int argc, char** argv ) -> int {
  auto input_result =
      parse_input_file( { argv, static_cast<std::size_t>( argc ) } );
  if ( !input_result ) {
    std::println( "Error: {}", input_result.error( ) );
    return AthelasExitCodes::FAILURE;
    ;
  }

  std::string input_path = *input_result;

  auto sig1 = signal( SIGSEGV, segfault_handler );
  auto sig2 = signal( SIGABRT, segfault_handler );

  // create span of args
  auto args = std::span( argv, static_cast<size_t>( argc ) );

  Kokkos::initialize( argc, argv );
  {
    // pin
    const auto pin = std::make_unique<ProblemIn>( input_path );

    // --- Create Driver ---
    Driver driver( pin.get( ) );

    // --- Timer ---
    Kokkos::Timer timer_total;

    // --- execute driver ---
    driver.execute( );

    // --- Finalize timer ---
    double const time = timer_total.seconds( );
    std::println( " ~ Done! Elapsed time: {} seconds.", time );
  }
  Kokkos::finalize( );

  return AthelasExitCodes::SUCCESS;
} // main
