/**
 * File    :  Initialization.cpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Initialize conserved fields for given problem
 **/

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "Kokkos_Core.hpp"

#include "Constants.h"
#include "Error.h"
#include "Grid.h"
#include "Initialization.h"

/**
 * Initialize the conserved Fields for various problems.
 * TODO: For now I initialize constant on each cell. Is there a better way?
 * TODO: iNodeX and order separation
 **/
void InitializeFields( Kokkos::View<double***> uCF, Kokkos::View<double***> uPF,
                       GridStructure& Grid, const unsigned int pOrder,
                       const double GAMMA_IDEAL, const std::string ProblemName )
{

  const unsigned int ilo    = Grid.Get_ilo( );
  const unsigned int ihi    = Grid.Get_ihi( );
  const unsigned int nNodes = Grid.Get_nNodes( );

  const unsigned int iCF_Tau = 0;
  const unsigned int iCF_V   = 1;
  const unsigned int iCF_E   = 2;

  const unsigned int iPF_D = 0;

  if ( ProblemName == "Sod" )
  {
    const double V0  = 0.0;
    const double D_L = 1.0;
    const double D_R = 0.125;
    const double P_L = 1.0;
    const double P_R = 0.1;

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < pOrder; k++ )
        for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ )
        {
          X1                    = Grid.Get_Centers( iX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( X1 <= 0.5 )
          {
            if ( k == 0 )
            {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_L;
              uCF( iCF_V, iX, 0 )   = V0;
              uCF( iCF_E, iX, 0 )   = ( P_L / 0.4 ) * uCF( iCF_Tau, iX, 0 );
            }

            uPF( iPF_D, iX, iNodeX ) = D_L;
          }
          else
          { // right domain
            if ( k == 0 )
            {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_R;
              uCF( iCF_V, iX, 0 )   = V0;
              uCF( iCF_E, iX, 0 )   = ( P_R / 0.4 ) * uCF( iCF_Tau, iX, 0 );
            }

            uPF( iPF_D, iX, iNodeX ) = D_R;
          }
        }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  }
  else if ( ProblemName == "ShuOsher" )
  {
    const double V0  = 2.629369;
    const double D_L = 3.857143;
    const double P_L = 10.333333;
    const double P_R = 1.0;

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < pOrder; k++ )
        for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ )
        {
          X1                    = Grid.Get_Centers( iX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( X1 <= -4.0 )
          {
            if ( k == 0 )
            {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_L;
              uCF( iCF_V, iX, 0 )   = V0;
              uCF( iCF_E, iX, 0 ) =
                  ( P_L / 0.4 ) * uCF( iCF_Tau, iX, 0 ) + 0.5 * V0 * V0;
            }

            uPF( iPF_D, iX, iNodeX ) = D_L;
          }
          else
          { // right domain
            if ( k == 0 )
            {
              uCF( iCF_Tau, iX, 0 ) =
                  1.0 / ( 1.0 + 0.2 * sin( 5.0 * PI( ) * X1 ) );
              uCF( iCF_V, iX, 0 ) = 0.0;
              uCF( iCF_E, iX, 0 ) = ( P_R / 0.4 ) * uCF( iCF_Tau, iX, 0 );
            }

            uPF( iPF_D, iX, iNodeX ) = ( 1.0 + 0.2 * sin( 5.0 * PI( ) * X1 ) );
          }
        }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  }
  else if ( ProblemName == "MovingContact" )
  {
    // Moving Contact problem.
    const double V0  = 0.1;
    const double D_L = 1.4;
    const double D_R = 1.0;
    const double P_L = 1.0;
    const double P_R = 1.0;

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < pOrder; k++ )
        for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ )
        {
          X1                    = Grid.Get_Centers( iX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( X1 <= 0.5 )
          {
            if ( k == 0 )
            {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_L;
              uCF( iCF_V, iX, 0 )   = V0;
              uCF( iCF_E, iX, 0 ) =
                  ( P_L / 0.4 ) * uCF( iCF_Tau, iX, 0 ) + 0.5 * V0 * V0;
            }

            uPF( iPF_D, iX, iNodeX ) = D_L;
          }
          else
          {
            if ( k == 0 )
            {
              uCF( iCF_Tau, iX, k ) = 1.0 / D_R;
              uCF( iCF_V, iX, k )   = V0;
              uCF( iCF_E, iX, k ) =
                  ( P_R / 0.4 ) * uCF( iCF_Tau, iX, k ) + 0.5 * V0 * V0;
            }

            uPF( iPF_D, iX, iNodeX ) = D_R;
          }
        }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  }
  else if ( ProblemName == "SmoothAdvection" )
  {
    // Smooth advection problem
    const double V0  = 1.0;
    const double P0  = 0.01;
    const double Amp = 1.0;

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < pOrder; k++ )
        for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ )
        {
          X1 = Grid.Get_Centers( iX );

          if ( k != 0 )
          {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = 0.0;
            uCF( iCF_E, iX, k )   = 0.0;
          }
          else
          {
            uCF( iCF_Tau, iX, k ) =
                1.0 / ( 2.0 + Amp * sin( 2.0 * PI( ) * X1 ) );
            uCF( iCF_V, iX, k ) = V0;
            uCF( iCF_E, iX, k ) =
                ( P0 / 0.4 ) * uCF( iCF_Tau, iX, k ) + 0.5 * V0 * V0;
          }
          uPF( iPF_D, iX, iNodeX ) = ( 2.0 + Amp * sin( 2.0 * PI( ) * X1 ) );
        }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  }
  else if ( ProblemName == "Sedov" )
  {
    // Smooth advection problem
    const double V0 = 0.0;
    const double D0 = 1.0;
    const double E0 = 0.3;

    const int origin = Grid.Get_nElements( ) / 2;

    const double P0 = ( 5.0 / 3.0 - 1.0 ) * E0 / Grid.Get_Widths( origin );

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < pOrder; k++ )
        for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ )
        {
          X1 = Grid.Get_Centers( iX );

          if ( k != 0 )
          {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = 0.0;
            uCF( iCF_E, iX, k )   = 0.0;
          }
          else
          {
            uCF( iCF_Tau, iX, k ) = 1.0 / D0;
            uCF( iCF_V, iX, k )   = V0;
            if ( iX == origin || iX == origin + 1 )
            {
              uCF( iCF_E, iX, k ) =
                  ( P0 / ( 5.0 / 3.0 - 1.0 ) ) * uCF( iCF_Tau, iX, k ) +
                  0.5 * V0 * V0;
            }
            else
            {
              uCF( iCF_E, iX, k ) =
                  ( 1.0e-6 / ( 5.0 / 3.0 - 1.0 ) ) * uCF( iCF_Tau, iX, k ) +
                  0.5 * V0 * V0;
            }
          }
          uPF( iPF_D, iX, iNodeX ) = D0;
        }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  }
  else if ( ProblemName == "Noh" )
  {
    const double V_L = 1.0;
    const double D_L = 1.0;
    const double P_L = 0.000001;
    const double D_R = 1.0;
    const double V_R = -1.0;
    const double P_R = 0.000001;

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < pOrder; k++ )
        for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ )
        {
          X1                    = Grid.NodeCoordinate( iX, iNodeX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( X1 <= 0.5 )
          {
            if ( k == 0 )
            {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_L;
              uCF( iCF_V, iX, 0 )   = V_L;
              uCF( iCF_E, iX, 0 ) =
                  ( P_L / ( GAMMA_IDEAL - 1.0 ) ) * uCF( iCF_Tau, iX, 0 ) +
                  0.5 * V_L * V_L;
            }

            uPF( iPF_D, iX, iNodeX ) = D_L;
          }
          else
          { // right domain
            if ( k == 0 )
            {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_R;
              uCF( iCF_V, iX, 0 )   = V_R;
              uCF( iCF_E, iX, 0 ) =
                  ( P_R / ( GAMMA_IDEAL - 1.0 ) ) * uCF( iCF_Tau, iX, 0 ) +
                  0.5 * V_R * V_R;
            }

            uPF( iPF_D, iX, iNodeX ) = D_R;
          }
        }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  }
  else if ( ProblemName == "ShocklessNoh" )
  {
    const double D   = 1.0;
    const double E_M = 1.0;
    const double P   = ( GAMMA_IDEAL - 1.0 ) * E_M * D;

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < pOrder; k++ )
        for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ )
        {
          X1                    = Grid.Get_Centers( iX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( k == 0 )
          {
            uCF( iCF_Tau, iX, 0 ) = 1.0 / D;
            uCF( iCF_V, iX, 0 )   = -X1;
            uCF( iCF_E, iX, 0 ) =
                E_M + 0.5 * uCF( iCF_V, iX, 0 ) * uCF( iCF_V, iX, 0 );
          }
          else if ( k == 1 )
          {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = -Grid.Get_Widths( iX );
            uCF( iCF_E, iX, k )   = ( -X1 ) * ( -Grid.Get_Widths( iX ) );
          }
          else if ( k == 2 )
          {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = 0.0;
            uCF( iCF_E, iX, k )   = uCF( iCF_V, iX, 1 ) * uCF( iCF_V, iX, 1 );
          }
          else
          {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = 0.0;
            uCF( iCF_E, iX, k )   = 0.0;
          }

          uPF( iPF_D, iX, iNodeX ) = D;
        }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  }
  else if ( ProblemName == "SmoothFlow" )
  {

    double X1  = 0.0;
    double amp = 0.999999999999999999999999999999999995;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < pOrder; k++ )
        for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ )
        {
          X1                    = Grid.Get_Centers( iX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( k == 0 )
          {
            double D              = ( 1.0 + amp * sin( PI( ) * X1 ) );
            uCF( iCF_Tau, iX, 0 ) = 1.0 / D;
            uCF( iCF_V, iX, 0 )   = 0.0;
            uCF( iCF_E, iX, 0 )   = ( D * D * D / 0.4 ) * uCF( iCF_Tau, iX, 0 );
          }
          else if ( k == 1 )
          {
            double D  = ( 1.0 + amp * sin( PI( ) * X1 ) );
            double dD = ( amp * PI( ) * cos( PI( ) * X1 ) );
            uCF( iCF_Tau, iX, k ) =
                ( -1 / ( D * D ) ) * dD * Grid.Get_Widths( iX );
            uCF( iCF_V, iX, k ) = 0.0;
            uCF( iCF_E, iX, k ) =
                ( ( 2.0 / 0.4 ) * D ) * dD * Grid.Get_Widths( iX );
          }
          else if ( k == 2 )
          {
            double D   = ( 1.0 + amp * sin( PI( ) * X1 ) );
            double dD  = ( amp * PI( ) * cos( PI( ) * X1 ) );
            double ddD = -( amp * PI( ) * PI( ) ) * sin( PI( ) * X1 );
            uCF( iCF_Tau, iX, k ) = ( 2.0 / ( D * D * D ) ) * ddD *
                                    Grid.Get_Widths( iX ) *
                                    Grid.Get_Widths( iX );
            uCF( iCF_V, iX, k ) = 0.0;
            uCF( iCF_E, iX, k ) = ( 2.0 / 0.4 ) * ddD * Grid.Get_Widths( iX ) *
                                  Grid.Get_Widths( iX );
          }
          else if ( k == 3 )
          {
            double D    = ( 1.0 + amp * sin( PI( ) * X1 ) );
            double dD   = ( amp * PI( ) * cos( PI( ) * X1 ) );
            double ddD  = -( amp * PI( ) * PI( ) ) * sin( PI( ) * X1 );
            double dddD = -( amp * PI( ) * PI( ) * PI( ) ) * cos( PI( ) * X1 );
            uCF( iCF_Tau, iX, k ) =
                ( -6.0 / ( D * D * D * D ) ) * dddD * Grid.Get_Widths( iX ) *
                Grid.Get_Widths( iX ) * Grid.Get_Widths( iX );
            uCF( iCF_V, iX, k ) = 0.0;
            uCF( iCF_E, iX, k ) = 0.0;
          }
          else
          {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = 0.0;
            uCF( iCF_E, iX, k )   = 0.0;
          }

          uPF( iPF_D, iX, iNodeX ) = ( 1.0 + amp * sin( PI( ) * X1 ) );
        }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  }
  else
  {
    throw Error( "Please choose a valid ProblemName" );
  }
}
