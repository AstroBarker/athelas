#ifndef SLOPELIMITER_UTILITIES_H
#define SLOPELIMITER_UTILITIES_H

double minmod( double a, double b, double c );
double minmodB( double a, double b, double c, double dx, double M );
double BarthJespersen( double U_v_L, double U_v_R, double U_c_L, double U_c_T,
                       double U_c_R, double alpha );
#endif
