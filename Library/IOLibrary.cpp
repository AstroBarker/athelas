/**
 * File     :  FluidUtilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : HDF5 IO routines
**/ 

#include <iostream>
#include <string>
#include <vector>

#include "hdf5.h"
#include "H5Cpp.h"
#include "DataStructures.h"
#include "Grid.h"

#include "IOLibrary.h"

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
  const int ihi = Grid.Get_ihi();

  const H5std_string FILE_NAME( fn );
  const H5std_string DATASET_NAME("Grid");
  const int size = (nX + 0*nGuard) * nNodes; // dataset dimensions

  std::vector<DataType> tau(size);
  std::vector<DataType> vel(size);
  std::vector<DataType> eint(size);
  std::vector<DataType> grid(size);

  for ( unsigned int iX = nGuard; iX <= ihi; iX++ )
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    grid[(iX-nGuard) * nNodes + iN].x = Grid(iX, iN);
    tau[(iX-nGuard) * nNodes + iN].x  = uCF(0, iX, iN);
    vel[(iX-nGuard) * nNodes + iN].x  = uCF(1, iX, iN);
    eint[(iX-nGuard) * nNodes + iN].x = uCF(2, iX, iN);
  }

  // Tell HDF5 how to use my datatype
  // H5::CompType mtype_CF(sizeof(DataType));

  // Define the datatype to pass HDF5
  // mtype_grid.insertMember( "Grid", HOFFSET(DataType, x), H5::PredType::NATIVE_DOUBLE );
  // mtype_CF.insertMember( "Specific Volume", HOFFSET(DataType, x), H5::PredType::NATIVE_DOUBLE );
  // mtype_CF.insertMember( "Velocity", HOFFSET(DataType, y), H5::PredType::NATIVE_DOUBLE );
  // mtype_CF.insertMember( "Specific Internal Energy", HOFFSET(DataType, z), H5::PredType::NATIVE_DOUBLE );

  // preparation of a dataset and a file.
  hsize_t dim[1];
  dim[0] = tau.size();                   // using vector::size()
  const int rank = sizeof(dim) / sizeof(hsize_t);
  H5::DataSpace space(rank, dim);

  H5::H5File file( fn, H5F_ACC_TRUNC );
  // Groups
  H5::Group group_grid = file.createGroup("/Spatial Grid");
  H5::Group group_CF = file.createGroup("/Conserved Fields");

  //DataSets
  H5::DataSet dataset_grid( file.createDataSet("/Spatial Grid/Grid", H5::PredType::NATIVE_DOUBLE, space) );
  H5::DataSet dataset_tau( file.createDataSet("/Conserved Fields/Specific Volume", H5::PredType::NATIVE_DOUBLE, space) );
  H5::DataSet dataset_vel( file.createDataSet("/Conserved Fields/Velocity", H5::PredType::NATIVE_DOUBLE, space) );
  H5::DataSet dataset_eint( file.createDataSet("/Conserved Fields/Specific Internal Energy", H5::PredType::NATIVE_DOUBLE, space) );

  dataset_grid.write( grid.data(), H5::PredType::NATIVE_DOUBLE );
  dataset_tau.write( tau.data(), H5::PredType::NATIVE_DOUBLE );
  dataset_vel.write( vel.data(), H5::PredType::NATIVE_DOUBLE );
  dataset_eint.write( eint.data(), H5::PredType::NATIVE_DOUBLE );
  
  
}