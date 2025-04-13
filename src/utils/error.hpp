#ifndef ERROR_HPP_
#define ERROR_HPP_
/**
 * @file error.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Error handling
 */

#include <array>
#include <csignal> // For signal constants
#include <cstdio>
#include <exception>
#include <execinfo.h>
#include <iostream>
#include <mutex>
#include <print>
#include <sstream>
#include <stacktrace>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

#include "constants.hpp"
#include "state.hpp"

enum AthelasExitCodes {
  SUCCESS                       = 0,
  FAILURE                       = 1,
  PHYSICAL_CONSTRAINT_VIOLATION = 2,
  MEMORY_ERROR                  = 3,
  UNKNOWN_ERROR                 = 255
};


inline void print_backtrace( ) {
  std::cout << std::stacktrace::current() << std::endl;
}

[[noreturn]] inline void segfault_handler( int sig ) {
  std::println( stderr, "Received signal {}", sig );
  print_backtrace( );
  std::quick_exit( AthelasExitCodes::FAILURE );
}

class AthelasError : public std::exception {
 private:
  std::string m_message;
  std::string m_function;
  std::string m_file;
  int m_line;

 public:
  // Constructor with detailed error information
  explicit AthelasError( std::string message, const std::string& function = "",
                         const std::string& file = "", int line = 0 )
      : m_message( std::move( message ) ), m_function( function ),
        m_file( file ), m_line( line ) {}

  // Override what() to provide error details
  [[nodiscard]] auto what( ) const noexcept -> const char* override {
    static thread_local std::string full_message;
    std::ostringstream oss;

    oss << "!!! Athelas Error: " << m_message << "\n";

    if ( !m_function.empty( ) ) {
      oss << "In function: " << m_function << "\n";
    }

    if ( !m_file.empty( ) && m_line > 0 ) {
      oss << "Location: " << m_file << ":" << m_line << "\n";
    }

    full_message = oss.str( );
    return full_message.c_str( );
  }

};

template <typename... Args>
[[noreturn]] constexpr void THROW_ATHELAS_ERROR(
    const char* message, const char* function = __builtin_FUNCTION( ),
    const char* file = __builtin_FILE( ), int line = __builtin_LINE( ) ) {
  throw AthelasError( message, function, file, line );
}

template <typename T>
void check_state( T state, const int ihi, const bool do_rad ) {
  auto uCR     = state->Get_uCR( );
  auto uCF     = state->Get_uCF( );
  const Real c = constants::c_cgs;

  // Create host mirrors of the views
  auto uCR_h = Kokkos::create_mirror_view( uCR );
  auto uCF_h = Kokkos::create_mirror_view( uCF );

  // Copy data to host
  Kokkos::deep_copy( uCR_h, uCR );
  Kokkos::deep_copy( uCF_h, uCF );

  // Check state on host
  for ( int iX = 1; iX <= ihi; iX++ ) {

    const Real tau   = uCF_h( 0, iX, 0 ); // cell averages checked
    const Real vel   = uCF_h( 1, iX, 0 );
    const Real e_m   = uCF_h( 2, iX, 0 );
    const Real e_rad = uCR_h( 0, iX, 0 );
    const Real f_rad = uCR_h( 1, iX, 0 );

    if ( tau <= 0.0 ) {
      THROW_ATHELAS_ERROR( "Negative or zero density!" );
    }
    if ( std::isnan( tau ) ) {
      THROW_ATHELAS_ERROR( "Specific volume NaN!" );
    }

    if ( std::fabs( vel ) >= c ) {
      THROW_ATHELAS_ERROR( "Velocity reached or exceeded speed of light!" );
    }
    if ( std::isnan( vel ) ) {
      THROW_ATHELAS_ERROR( "Velocity NaN!" );
    }

    if ( e_m <= 0.0 ) {
      THROW_ATHELAS_ERROR( "Negative or zero specific total energy!" );
    }
    if ( std::isnan( e_m ) ) {
      THROW_ATHELAS_ERROR( "Specific energy NaN!" );
    }

    if ( do_rad ) {
      if ( std::isnan( e_rad ) ) {
        THROW_ATHELAS_ERROR( "Radiation energy NaN!" );
      }
      if ( e_rad <= 0.0 ) {
        THROW_ATHELAS_ERROR( "Negative or zero radiation energy density!" );
      }

      if ( std::isnan( f_rad ) ) {
        THROW_ATHELAS_ERROR( "Radiation flux NaN!" );
      }
    }
  }
}

#endif // ERROR_HPP_
