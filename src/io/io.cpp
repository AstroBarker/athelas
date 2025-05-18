/**
 * @file io.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief HDF5 and std out IO routines
 *
 * @details Collection of functions for IO using H5Cpp for HDF5 operations
 */

#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <print>
#include <string>
#include <vector>

#include "H5Cpp.h"

#include "grid.hpp"
#include "io.hpp"
#include "limiters/slope_limiter.hpp"
#include "polynomial_basis.hpp"

/**
 * Write to standard output some initialization info
 * for the current simulation.
 **/
void print_simulation_parameters( GridStructure grid, ProblemIn* pin,
                                  const Real CFL ) {
  const int nX        = grid.get_n_elements( );
  const int nNodes    = grid.get_n_nodes( );
  const int basis_int = ( pin->basis == poly_basis::legendre ) ? 0 : 1;

  std::println( " ~ --- Physics Parameters --- " );
  std::println( " ~ Radiation      : {}", static_cast<int>( pin->do_rad ) );
  std::println( "" );

  std::println( " ~ --- Order Parameters --- " );
  std::println( " ~ basis          : {} ( 0 : legendre, 1: taylor )",
                basis_int );
  std::println( " ~ Spatial Order  : {}", pin->pOrder );
  std::println( " ~ Temporal Order : {}", pin->tOrder );
  std::println( " ~ RK Stages      : {}", pin->nStages );
  std::println( "" );

  std::println( " ~ --- grid Parameters --- " );
  std::println( " ~ Mesh Elements  : {}", nX );
  std::println( " ~ Number Nodes   : {}", nNodes );
  std::println( " ~ Lower Boundary : {}", grid.get_x_l( ) );
  std::println( " ~ Upper Boundary : {}", grid.get_x_r( ) );
  std::println( "" );

  std::println( " ~ --- Limiter Parameters --- " );
  if ( pin->pOrder == 1 ) {
    std::println( " ~ Spatial Order 1: Slope limiter not applied." );
  } else {
    std::println( " ~ gamma_l          : {}", pin->gamma_l );
    std::println( " ~ gamma_i          : {}", pin->gamma_i );
    std::println( " ~ gamma_r          : {}", pin->gamma_r );
    std::println( " ~ weno_r           : {}", pin->weno_r );
  }
  if ( pin->TCI_Option ) {
    std::println( " ~ TCI Value      : {}", pin->TCI_Threshold );
  } else {
    std::println( " ~ TCI Not Used." );
  }
  if ( pin->Characteristic ) {
    std::println( " ~ Limiting       : Characteristic" );
  } else {
    std::println( " ~ Limiting       : Componentwise" );
  }
  std::println( "" );

  std::println( " ~ --- Other --- " );
  std::cout << " ~ problem_name    : " << pin->problem_name << std::endl;
  std::println( " ~ CFL            : {}", CFL );
  std::println( "" );
}

/**
 * Write simulation output to disk
 **/
void write_state( State* state, GridStructure grid, SlopeLimiter* SL,
                  const std::string& problem_name, Real time, int order,
                  int i_write, bool do_rad ) {
  View3D<Real> uCF = state->get_u_cf( );
  View3D<Real> uCR = state->get_u_cr( );
  View3D<Real> uPF = state->get_u_pf( );

  // Construct filename
  std::string fn = problem_name;
  auto i_str     = std::to_string( i_write );
  int n_pad      = 0;
  if ( i_write < 10 ) {
    n_pad = 4;
  } else if ( i_write >= 10 && i_write < 100 ) {
    n_pad = 3;
  } else if ( i_write >= 100 && i_write < 1000 ) {
    n_pad = 2;
  } else if ( i_write >= 1000 && i_write < 10000 ) {
    n_pad = 1;
  }
  std::string suffix = std::string( n_pad, '0' ).append( i_str );
  fn.append( "_" );
  if ( i_write != -1 ) {
    fn.append( suffix );
  } else {
    fn.append( "final" );
  }
  fn.append( ".h5" );

  const char* fn2 = fn.c_str( );

  const int nX   = grid.get_n_elements( );
  const int ilo  = grid.get_ilo( );
  const int ihi  = grid.get_ihi( );
  const int size = ( nX * order );

  // Create data vectors
  std::vector<DataType> tau( size );
  std::vector<DataType> vel( size );
  std::vector<DataType> eint( size );
  std::vector<DataType> erad( size );
  std::vector<DataType> frad( size );
  std::vector<DataType> gridout( nX );
  std::vector<DataType> dr( nX );
  std::vector<DataType> limiter( nX );

  // Fill data vectors
  for ( int k = 0; k < order; k++ ) {
    for ( int iX = ilo; iX <= ihi; iX++ ) {
      gridout[( iX - ilo )].x           = grid.get_centers( iX );
      dr[( iX - ilo )].x                = grid.get_widths( iX );
      limiter[( iX - ilo )].x           = get_limited( SL, iX );
      tau[( iX - ilo ) + ( k * nX )].x  = uCF( 0, iX, k );
      vel[( iX - ilo ) + ( k * nX )].x  = uCF( 1, iX, k );
      eint[( iX - ilo ) + ( k * nX )].x = uCF( 2, iX, k );
      if ( do_rad ) {
        erad[( iX - ilo ) + ( k * nX )].x = uCR( 0, iX, k );
        frad[( iX - ilo ) + ( k * nX )].x = uCR( 1, iX, k );
      }
    }
  }

  // Create HDF5 file and datasets
  H5::H5File const file( fn2, H5F_ACC_TRUNC );

  // Create groups
  H5::Group const group_md   = file.createGroup( "/metadata" );
  H5::Group const group_mdb  = file.createGroup( "/metadata/build" );
  H5::Group const group_grid = file.createGroup( "/grid" );
  H5::Group const group_CF   = file.createGroup( "/conserved" );
  H5::Group const group_DF   = file.createGroup( "/diagnostic" );

  // Create datasets with proper dimensions
  std::array<hsize_t, 1> dim = { static_cast<hsize_t>( tau.size( ) ) };
  H5::DataSpace space( 1, dim.data( ) );

  std::array<hsize_t, 1> dim_grid = { static_cast<hsize_t>( gridout.size( ) ) };
  H5::DataSpace space_grid( 1, dim_grid.data( ) );

  std::array<hsize_t, 1> dim_md = { 1 };
  H5::DataSpace md_space( 1, dim_md.data( ) );

  // --- build info ---
  H5::StrType stringtype( H5::PredType::C_S1, H5T_VARIABLE );
  H5::DataSet const dataset_ghash =
      file.createDataSet( "/metadata/build/git_hash", stringtype, md_space );
  H5::DataSet const dataset_compiler =
      file.createDataSet( "/metadata/build/compiler", stringtype, md_space );
  H5::DataSet const dataset_timestamp =
      file.createDataSet( "/metadata/build/timestamp", stringtype, md_space );
  H5::DataSet const dataset_arch =
      file.createDataSet( "/metadata/build/arch", stringtype, md_space );
  H5::DataSet const dataset_os =
      file.createDataSet( "/metadata/build/os", stringtype, md_space );

  dataset_ghash.write( build_info::GIT_HASH, stringtype );
  dataset_compiler.write( build_info::COMPILER, stringtype );
  dataset_timestamp.write( build_info::BUILD_TIMESTAMP, stringtype );
  dataset_arch.write( build_info::ARCH, stringtype );
  dataset_os.write( build_info::OS, stringtype );

  // --- Create and write datasets ---
  H5::DataSet const dataset_nx =
      file.createDataSet( "/metadata/nx", H5::PredType::NATIVE_INT, md_space );
  H5::DataSet const dataset_order = file.createDataSet(
      "/metadata/order", H5::PredType::NATIVE_INT, md_space );
  H5::DataSet const dataset_time = file.createDataSet(
      "/metadata/time", H5::PredType::NATIVE_DOUBLE, md_space );
  H5::DataSet dataset_grid =
      file.createDataSet( "/grid/x", H5::PredType::NATIVE_DOUBLE, space_grid );
  H5::DataSet dataset_width =
      file.createDataSet( "/grid/dx", H5::PredType::NATIVE_DOUBLE, space_grid );
  H5::DataSet dataset_tau = file.createDataSet(
      "/conserved/tau", H5::PredType::NATIVE_DOUBLE, space );
  H5::DataSet dataset_vel = file.createDataSet(
      "/conserved/velocity", H5::PredType::NATIVE_DOUBLE, space );
  H5::DataSet dataset_eint = file.createDataSet(
      "/conserved/energy", H5::PredType::NATIVE_DOUBLE, space );

  H5::DataSet dataset_limiter = file.createDataSet(
      "/diagnostic/limiter", H5::PredType::NATIVE_DOUBLE, space_grid );

  // Write data
  dataset_nx.write( &nX, H5::PredType::NATIVE_INT );
  dataset_order.write( &order, H5::PredType::NATIVE_INT );
  dataset_time.write( &time, H5::PredType::NATIVE_DOUBLE );

  dataset_grid.write( gridout.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_width.write( dr.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_limiter.write( limiter.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_tau.write( tau.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_vel.write( vel.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_eint.write( eint.data( ), H5::PredType::NATIVE_DOUBLE );

  if ( do_rad ) {
    H5::DataSet dataset_erad = file.createDataSet(
        "/conserved/rad_energy", H5::PredType::NATIVE_DOUBLE, space );
    H5::DataSet dataset_frad = file.createDataSet(
        "/conserved/rad_momentum", H5::PredType::NATIVE_DOUBLE, space );
    dataset_erad.write( erad.data( ), H5::PredType::NATIVE_DOUBLE );
    dataset_frad.write( frad.data( ), H5::PredType::NATIVE_DOUBLE );
  }
}

/**
 * Write Modal basis coefficients and mass matrix
 **/
void write_basis( ModalBasis* basis, unsigned int ilo, unsigned int ihi,
                  unsigned int nNodes, unsigned int order,
                  const std::string& problem_name ) {
  std::string fn = problem_name;
  fn.append( "_basis" );
  fn.append( ".h5" );

  const char* fn2 = fn.c_str( );

  // Calculate total size needed
  const size_t total_size = static_cast<size_t>( ihi ) * ( nNodes + 2 ) * order;

  // Use std::vector instead of raw pointer for automatic memory management
  std::vector<Real> data( total_size );

  // Fill data using vector indexing instead of pointer arithmetic
  for ( unsigned int iX = ilo; iX <= ihi; iX++ ) {
    for ( unsigned int iN = 0; iN < nNodes + 2; iN++ ) {
      for ( unsigned int k = 0; k < order; k++ ) {
        const size_t idx =
            ( ( ( iX - ilo ) * ( nNodes + 2 ) + iN ) * order ) + k;
        data[idx] =
            basis->get_phi( static_cast<int>( iX ), static_cast<int>( iN ),
                            static_cast<int>( k ) );
      }
    }
  }

  // Create HDF5 file and dataset
  H5::H5File const file( fn2, H5F_ACC_TRUNC );
  std::array<hsize_t, 3> dimsf = { static_cast<hsize_t>( ihi ),
                                   static_cast<hsize_t>( nNodes + 2 ),
                                   static_cast<hsize_t>( order ) };
  H5::DataSpace dataspace( 3, dimsf.data( ) );
  H5::DataSet basisDataset =
      file.createDataSet( "basis", H5::PredType::NATIVE_DOUBLE, dataspace );

  // Write to File
  basisDataset.write( data.data( ), H5::PredType::NATIVE_DOUBLE );
}
