/**
 * File     :  IOLibrary.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : HDF5 IO routines
**/ 

#include <iostream>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "DataStructures.h"
#include "Grid.h"
#include "PolynomialBasis.h"
#include "SlopeLimiter.h"

#include "IOLibrary.h"


/**
 * Write to standard output some initialization info
 * for the current simulation.
**/
void PrintSimulationParameters( GridStructure& Grid, unsigned int pOrder, 
  unsigned int tOrder, unsigned int nStages, double CFL, double alpha, 
  double TCI, bool Char_option, bool TCI_Option, 
  std::string ProblemName )
{
  const unsigned int nX = Grid.Get_nElements();
  const unsigned int nNodes = Grid.Get_nNodes();

  std::printf("--- Order Parameters --- \n");
  std::printf("Spatial Order  : %d\n", pOrder);
  std::printf("Temporal Order : %d\n", tOrder);
  std::printf("RK Stages      : %d\n", nStages);
  std::printf("\n");

  std::printf("--- Grid Parameters --- \n");
  std::printf("Mesh Elements  : %d\n", nX);
  std::printf("Number Nodes   : %d\n", nNodes);
  std::printf("Lower Boudnary : %f\n", Grid.Get_xL());
  std::printf("Upper Boudnary : %f\n", Grid.Get_xR());
  std::printf("\n");

  std::printf("--- Limiter Parameters --- \n");
  if ( pOrder == 1)
  {
    printf("Spatial Order 1: Slope limiter not applied.\n");
  }
  else
  {
    std::printf("Alpha          : %f\n", alpha);
  }
  if ( TCI_Option )
  {
    std::printf("TCI Value      : %f\n", TCI);
  }
  else
  {
    std::printf("TCI Not Used.\n");
  }
  if ( Char_option )
  {
    std::printf("Limiting       : Characteristic \n");
  }
  else
  {
    std::printf("Limiting       : Componentwise\n");
  }
  std::printf("\n");

  std::printf("--- Other --- \n");
  std::cout << "ProblemName    : " << ProblemName << std::endl;
  std::printf("CFL            : %f\n", CFL);
  std::printf("\n");
}


//TODO: add Time
void WriteState( DataStructure3D& uCF, DataStructure3D& uPF, 
  DataStructure3D& uAF, GridStructure& Grid, SlopeLimiter& SL, 
  const std::string ProblemName )
{

  std::string fn = "athelas_";
  fn.append( ProblemName );
  fn.append( ".h5" );

  const char * fn2 = fn.c_str();

  const unsigned int nX     = Grid.Get_nElements();
  const unsigned int nGuard = Grid.Get_Guard();
  const unsigned int ilo = Grid.Get_ilo();
  const unsigned int ihi = Grid.Get_ihi();

  const H5std_string FILE_NAME( fn );
  const H5std_string DATASET_NAME("Grid");
  const int size = (nX);// * nNodes; // dataset dimensions

  std::vector<DataType> tau(size);
  std::vector<DataType> vel(size);
  std::vector<DataType> eint(size);
  std::vector<DataType> grid(size);
  std::vector<DataType> limiter(size);

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {
    grid[(iX-ilo)].x = Grid.Get_Centers(iX);
    limiter[(iX-ilo)].x = SL.Get_Limited(iX);
    tau[(iX-ilo)].x  = uCF(0, iX, 0);
    vel[(iX-ilo)].x  = uCF(1, iX, 0);
    eint[(iX-ilo)].x = uCF(2, iX, 0);
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

  H5::H5File file( fn2, H5F_ACC_TRUNC );
  // Groups
  H5::Group group_grid = file.createGroup("/Spatial Grid");
  H5::Group group_CF = file.createGroup("/Conserved Fields");
  H5::Group group_LF = file.createGroup("/Diagnostic Fields");

  //DataSets
  H5::DataSet dataset_grid( file.createDataSet("/Spatial Grid/Grid", H5::PredType::NATIVE_DOUBLE, space) );
  H5::DataSet dataset_tau( file.createDataSet("/Conserved Fields/Specific Volume", H5::PredType::NATIVE_DOUBLE, space) );
  H5::DataSet dataset_vel( file.createDataSet("/Conserved Fields/Velocity", H5::PredType::NATIVE_DOUBLE, space) );
  H5::DataSet dataset_eint( file.createDataSet("/Conserved Fields/Specific Internal Energy", H5::PredType::NATIVE_DOUBLE, space) );

  H5::DataSet dataset_limiter( file.createDataSet("/Diagnostic Fields/Limiter", H5::PredType::NATIVE_DOUBLE, space) );

  dataset_grid.write( grid.data(), H5::PredType::NATIVE_DOUBLE );
  dataset_limiter.write( limiter.data(), H5::PredType::NATIVE_DOUBLE );
  dataset_tau.write( tau.data(), H5::PredType::NATIVE_DOUBLE );
  dataset_vel.write( vel.data(), H5::PredType::NATIVE_DOUBLE );
  dataset_eint.write( eint.data(), H5::PredType::NATIVE_DOUBLE );
  
}


/**
 * Write Modal Basis coefficients and mass matrix
**/
void WriteBasis( ModalBasis& Basis, unsigned int ilo, 
  unsigned int ihi, unsigned int nNodes, unsigned int order, 
  std::string ProblemName )
{
  std::string fn = "athelas_basis_";
  fn.append( ProblemName );
  fn.append( ".h5" );

  const char * fn2 = fn.c_str();

  const H5std_string FILE_NAME( fn );
  const H5std_string DATASET_NAME("Basis");

  double* data = new double[ ihi*(nNodes+2)*order ];
  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  for ( unsigned int iN = 0; iN < nNodes+2; iN++ )
  for ( unsigned int k = 0; k < order; k++ )
  {
    data[((iX-ilo) * (nNodes+2) + iN) * order + k] = Basis.Get_Phi(iX,iN,k);
  }

  // Create HDF5 file and dataset
  H5::H5File file( fn2, H5F_ACC_TRUNC );
  hsize_t dimsf[3] = {ihi, nNodes+2, order};
  H5::DataSpace dataspace(3, dimsf);
  H5::DataSet BasisDataset( file.createDataSet("Basis", H5::PredType::NATIVE_DOUBLE,
                                          dataspace) );
  // Write to File
  BasisDataset.write( data, H5::PredType::NATIVE_DOUBLE );

  delete [] data;
}