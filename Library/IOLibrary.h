#ifndef IOLIBRARY_H
#define IOLIBRARY_H

struct GridType
{
  double r{};
};

struct DataType
{
    double x{};
};

void WriteState( DataStructure3D& uCF, DataStructure3D& uPF, 
  DataStructure3D& uAF, GridStructure& Grid, const std::string ProblemName );

void PrintSimulationParameters( GridStructure& Grid, unsigned int pOrder, 
  unsigned int tOrder, unsigned int nStages, double CFL, double Beta_TVD, 
  double Beta_TVB, double TCI, bool Char_option, bool TCI_Option, 
  std::string ProblemName );

void WriteBasis( ModalBasis& Basis, unsigned int ilo, 
  unsigned int ihi, unsigned int nNodes, unsigned int order, 
  std::string ProblemName );

#endif