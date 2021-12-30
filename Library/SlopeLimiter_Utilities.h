#ifndef SLOPELIMITER_UTILITIES_H
#define SLOPELIMITER_UTILITIES_H

double minmod( double a, double b, double c );
double minmodB( double a, double b, double c, double dx, double M );
double BarthJespersen( double* U_vertex, double* U_c, double alpha );
#endif
