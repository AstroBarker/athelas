/**
 * IO Routines
 **/

#include <iostream>
#include <string>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
// #include <assert.h>

#include "hdf5.h"
#include "H5Cpp.h"
// #include <hdf5.h>
// #include <hdf5_hl.h>
#include "DataStructures.h"
#include "Grid.h"

#include "IOLibrary.h"

// using namespace H5;

// Okay... need a way to do this.
// Need to unfurl my uCF etc DataStructure3D and write out individual arrays.
// I'd like to not allocate temporary arrays for this.
// But perhaps it is not so much of an issue.

//TODO: add Time
void WriteState( DataStructure3D& uCF, DataStructure3D& uPF, 
  DataStructure3D& uAF, GridStructure& Grid, const std::string ProblemName )
{

  std::string fn = "Splode_";
  fn.append( ProblemName );
  fn.append( ".h5" );

  const int nX     = Grid.Get_nElements();
  const int nNodes = Grid.Get_nNodes();
  const int nGuard = Grid.Get_Guard();

  const H5std_string FILE_NAME( fn );
  const H5std_string DATASET_NAME("Grid");
  const int size = (nX + 2*nGuard) * nNodes; // dataset dimensions

  double* tmp1 = new double[size];
  double* tmp2 = new double[size];
  double* tmp3 = new double[size];
  Grid.copy( tmp1 );

  // Create HDF5 file and dataset
  H5::H5File file( FILE_NAME, H5F_ACC_TRUNC );
  hsize_t dimsf[1] = {static_cast<hsize_t>( size )};
  H5::DataSpace dataspace(1, dimsf);
  H5::DataSet dataset = file.createDataSet( DATASET_NAME, 
    H5::PredType::NATIVE_DOUBLE, dataspace );
  // data to HDF5 file
  dataset.write( tmp1, H5::PredType::NATIVE_DOUBLE );

  // === Grid writen ===
  // may need better long term solution here.

  // double* tmp_big = new double[size * 3];
  H5::DataSet ds_tau = file.createDataSet( "Specific Volume", 
    H5::PredType::NATIVE_DOUBLE, dataspace );
  H5::DataSet ds_vel = file.createDataSet( "Velocity", 
    H5::PredType::NATIVE_DOUBLE, dataspace );
  H5::DataSet ds_int = file.createDataSet( "Specific Internal Energy", 
    H5::PredType::NATIVE_DOUBLE, dataspace );

  for ( unsigned int iX = nGuard; iX <= nX - nGuard + 1; iX++)
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    tmp1[iX * nNodes + iN] = uCF( 0, iX, iN );
    tmp2[iX * nNodes + iN] = uCF( 1, iX, iN );
    tmp3[iX * nNodes + iN] = uCF( 2, iX, iN );
  }
  ds_tau.write( tmp1, H5::PredType::NATIVE_DOUBLE );
  ds_vel.write( tmp2, H5::PredType::NATIVE_DOUBLE );
  ds_int.write( tmp3, H5::PredType::NATIVE_DOUBLE );



  delete [] tmp1;
  delete [] tmp2;
  delete [] tmp3;
}