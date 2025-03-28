/**
 * File    :  error.hpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Error throwing class, state checking
 **/

#ifndef ERROR_HPP_
#define ERROR_HPP_

#include <csignal> // For signal constants
#include <cstdio>
#include <exception>
#include <execinfo.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "constants.hpp"
#include "state.hpp"

inline void print_backtrace( ) {
  void *callstack[128];
  int frames     = backtrace( callstack, 128 );
  char **symbols = backtrace_symbols( callstack, frames );

  fprintf( stderr, "Backtrace:\n" );
  for ( int i = 0; i < frames; ++i ) {
    fprintf( stderr, "%s\n", symbols[i] );
  }

  free( symbols );
}

inline void segfault_handler( int sig ) {
  fprintf( stderr, "Received signal %d\n", sig );
  print_backtrace( );
  exit( 1 );
}

enum AthelasExitCodes {
  SUCCESS                       = 0,
  FAILURE                       = 1,
  PHYSICAL_CONSTRAINT_VIOLATION = 2,
  MEMORY_ERROR                  = 3,
  UNKNOWN_ERROR                 = 255
};

class AthelasError : public std::exception {
 private:
  std::string m_message;
  std::string m_function;
  std::string m_file;
  int m_line;

 public:
  // Constructor with detailed error information
  AthelasError( const std::string &message, const std::string &function = "",
                const std::string &file = "", int line = 0 )
      : m_message( message ), m_function( function ), m_file( file ),
        m_line( line ) {}

  // Override what() to provide error details
  const char *what( ) const noexcept override {
    static thread_local std::string full_message;
    std::ostringstream oss;

    oss << " ! Athelas Error: " << m_message << "\n";

    if ( !m_function.empty( ) ) {
      oss << "In function: " << m_function << "\n";
    }

    if ( !m_file.empty( ) && m_line > 0 ) {
      oss << "Location: " << m_file << ":" << m_line << "\n";
    }

    full_message = oss.str( );
    return full_message.c_str( );
  }

  // Destructor
  ~AthelasError( ) noexcept override {}
};

// Macro to simplify error throwing with file and line information
#define THROW_ATHELAS_ERROR( message )                                         \
  throw AthelasError( message, __FUNCTION__, __FILE__, __LINE__ )

template <typename T>
void check_state( T state, const int ihi, const bool do_rad ) {

  auto uCR = state->Get_uCR( );
  auto uCF = state->Get_uCF( );

  const Real c = constants::c_cgs;

  Kokkos::parallel_for(
      "check_state", ihi, KOKKOS_LAMBDA( const int i ) {
        const int iX = i + 1; // hack
                              //
        const Real tau   = uCF( 0, iX, 0 ); // cell averages checked
        const Real vel   = uCF( 1, iX, 0 );
        const Real e_m   = uCF( 2, iX, 0 );
        const Real e_rad = uCR( 0, iX, 0 );
        const Real f_rad = uCR( 1, iX, 0 );

        assert( tau > 0.0 && " ! Check state :: Negative or zero density!" );
        assert( !std::isnan( tau ) &&
                " ! Check state :: Specific volume NaN!" );

        assert(
            std::fabs( vel ) < c &&
            " ! Check state :: Velocity reached or exceeded speed of light!" );
        assert( !std::isnan( vel ) && " ! Check state :: Velocity NaN!" );

        assert( e_m > 0.0 &&
                " Check state :: ! Negative or zero specific total energy!" );
        assert( !std::isnan( e_m ) &&
                " ! Check state :: Specific energy NaN!" );

        if ( do_rad ) {
          assert(
              e_rad > 0.0 &&
              " ! Check state :: Negative or zero radiation energy density!" );
          assert( !std::isnan( e_rad ) &&
                  " ! Check state :: Radiation energy NaN!" );

          // TODO: radiation flux bound
          // assert ( f_rad >= 0.0 && " Check state :: ! Negative or zero
          // radiation energy density!" );
          assert( !std::isnan( f_rad ) &&
                  " ! Check state :: Radiation flux NaN!" );
        }
      } );
}

#endif // ERROR_HPP_
