#ifndef DRIVER_H
#define DRIVER_H

int NumNodes( unsigned int order );

double ComputeCFL( double CFL, unsigned int order, unsigned int nStages,
                   unsigned int tOrder );
#endif