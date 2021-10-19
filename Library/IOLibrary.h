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

#endif