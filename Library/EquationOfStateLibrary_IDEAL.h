#ifndef EQUATIONOFSTATELIBRARY_IDEAL_H
#define EQUATIONOFSTATELIBRARY_IDEAL_H

double ComputePressureFromPrimitive_IDEAL( double Ev,
                                           double GAMMA = 1.4 );
double ComputePressureFromConserved_IDEAL( double Tau, double V, double Em_T,
                                           double GAMMA = 1.4 );
double ComputeSoundSpeedFromConserved_IDEAL( double Tau, double V, double Em_T,
                                             double GAMMA = 1.4 );

#endif