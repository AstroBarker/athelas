/**
 * @file io.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief HDF5 and std out IO routines
 *
 * @details Collection of functions for IO
 */

#include <cstddef>
#include <iostream>
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
void PrintSimulationParameters( GridStructure Grid, ProblemIn* pin,
                                const Real CFL ) {
  const int nX     = Grid.Get_nElements( );
  const int nNodes = Grid.Get_nNodes( );
  const int basis_int = ( pin->Basis == PolyBasis::Legendre ) ? 0 : 1;

  std::println( " ~ --- Physics Parameters --- " );
  std::println( " ~ Radiation      : {}", static_cast<int>( pin->do_rad ) );
  std::println( "" );

  std::println( " ~ --- Order Parameters --- " );
  std::println( " ~ Basis          : {} ( 0 : Legendre, 1: Taylor )",
               basis_int );
  std::println( " ~ Spatial Order  : {}", pin->pOrder );
  std::println( " ~ Temporal Order : {}", pin->tOrder );
  std::println( " ~ RK Stages      : {}", pin->nStages );
  std::println( "" );

  std::println( " ~ --- Grid Parameters --- " );
  std::println( " ~ Mesh Elements  : {}", nX );
  std::println( " ~ Number Nodes   : {}", nNodes );
  std::println( " ~ Lower Boundary : {}", Grid.Get_xL( ) );
  std::println( " ~ Upper Boundary : {}", Grid.Get_xR( ) );
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
void WriteState( State* state, GridStructure Grid, SlopeLimiter* SL,
                 const std::string& problem_name, Real time, int order,
                 int i_write, bool do_rad ) {

  View3D<Real> uCF = state->Get_uCF( );
  View3D<Real> uCR = state->Get_uCR( );
  View3D<Real> uPF = state->Get_uPF( );

  std::string fn = "athelas_";
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
  } else {
    n_pad = 0;
  }
  std::string suffix = std::string( n_pad, '0' ).append( i_str );
  fn.append( problem_name );
  fn.append( "_" );
  if ( i_write != -1 ) {
    fn.append( suffix );
  } else {
    fn.append( "final" );
  }
  fn.append( ".h5" );

  // conversion to make HDF5 happy
  const char* fn2 = fn.c_str( );

  const int nX  = Grid.Get_nElements( );
  const int ilo = Grid.Get_ilo( );
  const int ihi = Grid.Get_ihi( );

  const int size = ( nX * order ); // dataset dimensions

  std::vector<DataType> tau( size );
  std::vector<DataType> vel( size );
  std::vector<DataType> eint( size );
  std::vector<DataType> erad( size );
  std::vector<DataType> frad( size );
  std::vector<DataType> grid( nX );
  std::vector<DataType> dr( nX );
  std::vector<DataType> limiter( nX );

  for ( int k = 0; k < order; k++ ) {
    for ( int iX = ilo; iX <= ihi; iX++ ) {
      grid[( iX - ilo )].x = Grid.Get_Centers( iX );
      dr[( iX - ilo )].x   = Grid.Get_Widths( iX );
      limiter[( iX - ilo )].x           = Get_Limited( SL, iX );
      tau[( iX - ilo ) + ( k * nX )].x  = uCF( 0, iX, k );
      vel[( iX - ilo ) + ( k * nX )].x  = uCF( 1, iX, k );
      eint[( iX - ilo ) + ( k * nX )].x = uCF( 2, iX, k );
      if ( do_rad ) {
        erad[( iX - ilo ) + ( k * nX )].x = uCR( 0, iX, k );
        frad[( iX - ilo ) + ( k * nX )].x = uCR( 1, iX, k );
      }
    }
  }

  // preparation of a dataset and a file.
  hsize_t dim[1];
  dim[0]         = tau.size( ); // using vector::size()
  const int rank = sizeof( dim ) / sizeof( hsize_t );
  H5::DataSpace const space( rank, dim );

  hsize_t dim_grid[1];
  dim_grid[0]         = grid.size( ); // using vector::size()
  const int rank_grid = sizeof( dim_grid ) / sizeof( hsize_t );
  H5::DataSpace const space_grid( rank_grid, dim_grid );

  hsize_t const len = 1;
  hsize_t dim_md[1] = { len };
  const int rank_md = 1;
  H5::DataSpace const md_space( rank_md, dim_md );

  H5::H5File const file( fn2, H5F_ACC_TRUNC );
  // Groups
  H5::Group const group_md   = file.createGroup( "/metadata" );
  H5::Group const group_grid = file.createGroup( "/grid" );
  H5::Group const group_CF   = file.createGroup( "/conserved" );
  H5::Group const group_DF   = file.createGroup( "/diagnostic" );

  // DataSets
  H5::DataSet const dataset_nx( file.createDataSet(
      "/metadata/nx", H5::PredType::NATIVE_INT, md_space ) );
  H5::DataSet const dataset_order( file.createDataSet(
      "/metadata/order", H5::PredType::NATIVE_INT, md_space ) );
  H5::DataSet const dataset_time( file.createDataSet(
      "/metadata/time", H5::PredType::NATIVE_DOUBLE, md_space ) );
  H5::DataSet dataset_grid( file.createDataSet(
      "/grid/x", H5::PredType::NATIVE_DOUBLE, space_grid ) );
  H5::DataSet dataset_width( file.createDataSet(
      "/grid/dx", H5::PredType::NATIVE_DOUBLE, space_grid ) );
  H5::DataSet dataset_tau( file.createDataSet(
      "/conserved/tau", H5::PredType::NATIVE_DOUBLE, space ) );
  H5::DataSet dataset_vel( file.createDataSet(
      "/conserved/velocity", H5::PredType::NATIVE_DOUBLE, space ) );
  H5::DataSet dataset_eint( file.createDataSet(
      "/conserved/energy", H5::PredType::NATIVE_DOUBLE, space ) );

  H5::DataSet dataset_limiter( file.createDataSet(
      "/diagnostic/limiter", H5::PredType::NATIVE_DOUBLE, space_grid ) );
  H5::DataSet dataset_erad( file.createDataSet(
      "/conserved/rad_energy", H5::PredType::NATIVE_DOUBLE, space ) );
  H5::DataSet dataset_frad( file.createDataSet(
      "/conserved/rad_momentum", H5::PredType::NATIVE_DOUBLE, space ) );

  // --- Write data ---
  dataset_nx.write( &nX, H5::PredType::NATIVE_INT );
  dataset_order.write( &order, H5::PredType::NATIVE_INT );
  dataset_time.write( &time, H5::PredType::NATIVE_DOUBLE );

  dataset_grid.write( grid.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_width.write( dr.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_limiter.write( limiter.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_tau.write( tau.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_vel.write( vel.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_eint.write( eint.data( ), H5::PredType::NATIVE_DOUBLE );

  if ( do_rad ) {
    dataset_erad.write( erad.data( ), H5::PredType::NATIVE_DOUBLE );
    dataset_frad.write( frad.data( ), H5::PredType::NATIVE_DOUBLE );
  }
}

/**
 * Write Modal Basis coefficients and mass matrix
 **/
void WriteBasis( ModalBasis* Basis, unsigned int ilo, unsigned int ihi,
                 unsigned int nNodes, unsigned int order,
                 const std::string& problem_name ) {
  std::string fn = "athelas_basis_";
  fn.append( problem_name );
  fn.append( ".h5" );

  const char* fn2 = fn.c_str( );

  Real* data =
      new Real[static_cast<unsigned long>( ihi * ( nNodes + 2 ) * order )];
  for ( unsigned int iX = ilo; iX <= ihi; iX++ ) {
    for ( unsigned int iN = 0; iN < nNodes + 2; iN++ ) {
      for ( unsigned int k = 0; k < order; k++ ) {
        data[( ( ( iX - ilo ) * ( nNodes + 2 ) + iN ) * order ) + k] =
            Basis->Get_Phi( iX, iN, k );
      }
    }
  }

  // Create HDF5 file and dataset
  H5::H5File const file( fn2, H5F_ACC_TRUNC );
  hsize_t dimsf[3] = { ihi, nNodes + 2, order };
  H5::DataSpace const dataspace( 3, dimsf );
  H5::DataSet const BasisDataset(
      file.createDataSet( "Basis", H5::PredType::NATIVE_DOUBLE, dataspace ) );
  // Write to File
  BasisDataset.write( data, H5::PredType::NATIVE_DOUBLE );

  delete[] data;
}
