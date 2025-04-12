/**
 * @file tableau.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Class for holding implicit and explicit RK tableaus.
 *
 * @details TODO: describe tableaus.
 */

#include "tableau.hpp"
#include "constants.hpp"

ButcherTableau::ButcherTableau( const int nStages_, const int tOrder_,
                                const TableauType type )
    : nStages( nStages_ ), tOrder( tOrder_ ),
      a_ij( "butcher a_ij", nStages_, nStages_ ),
      b_i( "butcher b_i", nStages_ ), type_( type ) {
  initialize_tableau( );
}
// Initialize arrays for timestepper
// TODO: Separate nStages from a tOrder
void ButcherTableau::initialize_tableau( ) {

  if ( tOrder == 1 and nStages > 1 ) {
    THROW_ATHELAS_ERROR( "\n \
      ! ButcherTableau :: Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. \n" );
  }
  if ( ( nStages != tOrder && nStages != 5 ) ) {
    THROW_ATHELAS_ERROR( "\n \
      ! ButcherTableau :: Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. \n" );
  }
  if ( ( tOrder == 4 && nStages != 5 ) ) {
    THROW_ATHELAS_ERROR( "\n \
      ! ButcherTableau :: Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. \n" );
  }
  if ( tOrder > 4 ) {
    THROW_ATHELAS_ERROR(
        "\n ! ButcherTableau :: Temporal torder > 4 not supported! \n" );
  }

  // Init to zero
  for ( int i = 0; i < nStages; i++ )
    for ( int j = 0; j < nStages; j++ ) {
      a_ij( i, j ) = 0.0;
      b_i( i )     = 0.0;
    }

  if ( type_ == TableauType::Explicit ) {
    // Forward Euler //
    if ( nStages == 1 and tOrder == 1 ) {
      a_ij( 0, 0 ) = 0.0;
      b_i( 0 )     = 1.0;
    } else if ( nStages == 2 && tOrder == 2 ) {
      a_ij( 1, 0 ) = 1.0;
      b_i( 0 )     = 0.5;
      b_i( 1 )     = 0.5;
    } else if ( nStages == 3 && tOrder == 3 ) {
      a_ij( 1, 0 ) = 1.0;
      a_ij( 2, 0 ) = 0.25;
      a_ij( 2, 1 ) = 0.25;
      b_i( 0 )     = 1.0 / 6.0;
      b_i( 1 )     = 1.0 / 6.0;
      b_i( 2 )     = 2.0 / 3.0;
      // (pex, pim, plin) = (2,2,5)
    } else if ( nStages == 5 && tOrder == 2 ) {
      a_ij( 1, 0 ) = 1.0;
      a_ij( 2, 0 ) = 0.99132899;
      a_ij( 3, 0 ) = 0.99130196;
      a_ij( 4, 0 ) = 0.99191257;
      a_ij( 2, 1 ) = 0.99132899;
      a_ij( 3, 1 ) = 0.96542648;
      a_ij( 4, 1 ) = 0.94453690;
      a_ij( 3, 2 ) = 0.97387092;
      a_ij( 4, 2 ) = 0.90549663;
      a_ij( 4, 3 ) = 0.92979121;
      b_i( 0 )     = 0.63253575;
      b_i( 1 )     = 0.25781844;
      b_i( 2 )     = 0.09173050;
      b_i( 3 )     = 0.00863176;
      b_i( 4 )     = 0.00928355;
    } else if ( nStages == 5 && tOrder == 3 ) {
      a_ij( 1, 0 ) = 1.0;
      a_ij( 2, 0 ) = 0.19736166;
      a_ij( 3, 0 ) = 0.06602780;
      a_ij( 4, 0 ) = 0.04161484;
      a_ij( 2, 1 ) = 0.19736166;
      a_ij( 3, 1 ) = 0.06602780;
      a_ij( 4, 1 ) = 0.02887068;
      a_ij( 3, 2 ) = 0.33455230;
      a_ij( 4, 2 ) = 0.14628314;
      a_ij( 4, 3 ) = 0.43725043;
      b_i( 0 )     = 0.15562497;
      b_i( 1 )     = 0.13677868;
      b_i( 2 )     = 0.29274344;
      b_i( 3 )     = 0.12620947;
      b_i( 4 )     = 0.28864344;
    } else if ( nStages == 5 && tOrder == 4 ) {
      a_ij( 1, 0 ) = 0.51047914;
      a_ij( 2, 0 ) = 0.08515080;
      a_ij( 3, 0 ) = 0.29902100;
      a_ij( 4, 0 ) = 0.01438455;
      a_ij( 2, 1 ) = 0.21940489;
      a_ij( 3, 1 ) = 0.07704762;
      a_ij( 4, 1 ) = 0.03706414;
      a_ij( 3, 2 ) = 0.46190055;
      a_ij( 4, 2 ) = 0.22219957;
      a_ij( 4, 3 ) = 0.63274729;
      b_i( 0 )     = 0.12051432;
      b_i( 1 )     = 0.22614012;
      b_i( 2 )     = 0.27630606;
      b_i( 3 )     = 0.12246455;
      b_i( 4 )     = 0.25457495;
    } else {
      THROW_ATHELAS_ERROR(
          " ! ButcherTableau :: Explicit :: Please choose a valid "
          "tableau! \n" );
    }
  }

  if ( type_ == TableauType::Implicit ) {

    // Backwards Euler //
    if ( nStages == 1 && tOrder == 1 ) {
      a_ij( 0, 0 ) = 1.0;
      b_i( 0 )     = 1.0;
    } else if ( nStages == 2 && tOrder == 2 ) {
      const static Real gam = 1.0 - 1.0 / std::sqrt( 2 );
      a_ij( 0, 0 )          = gam; // 0.71921758;
      a_ij( 1, 0 )          = 1.0 - gam; // 0.11776435;
      a_ij( 1, 1 )          = gam; // 0.16301806;
      b_i( 0 )              = 0.5;
      b_i( 1 )              = 0.5;
      // L-stable
    } else if ( nStages == 3 && tOrder == 3 ) {
      a_ij( 2, 0 ) = 1.0 / 6.0;
      a_ij( 1, 1 ) = 1.0;
      a_ij( 2, 1 ) = -1.0 / 3.0;
      a_ij( 2, 2 ) = 2.0 / 3.0;
      b_i( 0 )     = 1.0 / 6.0;
      b_i( 1 )     = 1.0 / 6.0;
      b_i( 2 )     = 2.0 / 3.0;
    } else if ( nStages == 5 && tOrder == 4 ) {
      a_ij( 0, 0 ) = 1.03217796e-16; // just 0?
      a_ij( 1, 0 ) = 0.510479144;
      a_ij( 2, 0 ) = 5.06048136e-3;
      a_ij( 3, 0 ) = 8.321807e-2;
      a_ij( 4, 0 ) = 7.56636565e-2;
      a_ij( 1, 1 ) = 1.00124199e-14;
      a_ij( 2, 1 ) = 1.00953283e-1;
      a_ij( 3, 1 ) = 1.60838280e-1;
      a_ij( 4, 1 ) = 1.25319139e-1;
      a_ij( 2, 2 ) = 1.98541931e-1;
      a_ij( 3, 2 ) = 3.28641063e-1;
      a_ij( 4, 2 ) = 7.08147871e-2;
      a_ij( 3, 3 ) = -3.84714236e-3;
      a_ij( 4, 3 ) = 6.34597980e-1;
      a_ij( 4, 4 ) = -7.22101223e-17;
      b_i( 0 )     = 0.12051432;
      b_i( 1 )     = 0.22614012;
      b_i( 2 )     = 0.27630606;
      b_i( 3 )     = 0.12246455;
      b_i( 4 )     = 0.25457495;
    } else {
      THROW_ATHELAS_ERROR( " ! ButcherTableau :: Implicit :: Please choose a "
                           "valid tableau! \n" );
    }

    // TODO: more tableaus
  }
}

ShuOsherTableau::ShuOsherTableau( const int nStages_, const int tOrder_,
                                  const TableauType type )
    : nStages( nStages_ ), tOrder( tOrder_ ),
      a_ij( "butcher a_ij", nStages_, nStages_ ),
      b_ij( "butcher b_i", nStages_, nStages_ ) {
  initialize_tableau( );
}

// Initialize arrays for timestepper
// TODO: Separate nStages from a tOrder
void ShuOsherTableau::initialize_tableau( ) {

  if ( tOrder == 1 and nStages > 1 ) {
    THROW_ATHELAS_ERROR( "\n \
      ! Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n" );
  }
  if ( ( nStages != tOrder && nStages != 5 ) ) {
    THROW_ATHELAS_ERROR( "\n \
      ! Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n" );
  }
  if ( ( tOrder == 4 && nStages != 5 ) ) {
    THROW_ATHELAS_ERROR( "\n \
      ! Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n" );
  }
  if ( tOrder > 4 ) {
    THROW_ATHELAS_ERROR( "\n ! Temporal torder > 4 not supported! \n" );
  }

  // Init to zero
  for ( int i = 0; i < nStages; i++ )
    for ( int j = 0; j < nStages; j++ ) {
      a_ij( i, j ) = 0.0;
      b_ij( i, j ) = 0.0;
    }

  if ( type_ == TableauType::Implicit ) {
    THROW_ATHELAS_ERROR(
        " ! ShuOsherTableau :: No implicit ShuOsher form tableaus "
        "implemented." );
  }

  if ( nStages < 5 ) {

    if ( tOrder == 1 ) {
      a_ij( 0, 0 ) = 1.0;
      b_ij( 0, 0 ) = 1.0;
    } else if ( tOrder == 2 ) {
      a_ij( 0, 0 ) = 1.0;
      a_ij( 1, 0 ) = 0.5;
      a_ij( 1, 1 ) = 0.5;

      b_ij( 0, 0 ) = 1.0;
      b_ij( 1, 0 ) = 0.0;
      b_ij( 1, 1 ) = 0.5;
    } else if ( tOrder == 3 ) {
      a_ij( 0, 0 ) = 1.0;
      a_ij( 1, 0 ) = 0.75;
      a_ij( 1, 1 ) = 0.25;
      a_ij( 2, 0 ) = 1.0 / 3.0;
      a_ij( 2, 1 ) = 0.0;
      a_ij( 2, 2 ) = 2.0 / 3.0;

      b_ij( 0, 0 ) = 1.0;
      b_ij( 1, 0 ) = 0.0;
      b_ij( 1, 1 ) = 0.25;
      b_ij( 2, 0 ) = 0.0;
      b_ij( 2, 1 ) = 0.0;
      b_ij( 2, 2 ) = 2.0 / 3.0;
    }
  } else if ( nStages == 5 ) {
    if ( tOrder == 1 ) {
      THROW_ATHELAS_ERROR( "\n ! We do support a 1st order, 5 stage SSPRK "
                           "integrator. \n" );
    } else if ( tOrder == 2 ) {
      a_ij( 0, 0 ) = 1.0;
      a_ij( 4, 0 ) = 0.2;
      a_ij( 1, 1 ) = 1.0;
      a_ij( 2, 2 ) = 1.0;
      a_ij( 3, 3 ) = 1.0;
      a_ij( 4, 4 ) = 0.8;

      b_ij( 0, 0 ) = 0.25;
      b_ij( 1, 1 ) = 0.25;
      b_ij( 2, 2 ) = 0.25;
      b_ij( 3, 3 ) = 0.25;
      b_ij( 4, 4 ) = 0.20;
    } else if ( tOrder == 3 ) {
      a_ij( 0, 0 ) = 1.0;
      a_ij( 1, 0 ) = 0.0;
      a_ij( 2, 0 ) = 0.56656131914033;
      a_ij( 3, 0 ) = 0.09299483444413;
      a_ij( 4, 0 ) = 0.00736132260920;
      a_ij( 1, 1 ) = 1.0;
      a_ij( 3, 1 ) = 0.00002090369620;
      a_ij( 4, 1 ) = 0.20127980325145;
      a_ij( 2, 2 ) = 0.43343868085967;
      a_ij( 4, 2 ) = 0.00182955389682;
      a_ij( 3, 3 ) = 0.90698426185967;
      a_ij( 4, 4 ) = 0.78952932024253;

      b_ij( 0, 0 ) = 0.37726891511710;
      b_ij( 3, 0 ) = 0.00071997378654;
      b_ij( 4, 0 ) = 0.00277719819460;
      b_ij( 1, 1 ) = 0.37726891511710;
      b_ij( 4, 1 ) = 0.00001567934613;
      b_ij( 2, 2 ) = 0.16352294089771;
      b_ij( 3, 3 ) = 0.34217696850008;
      b_ij( 4, 4 ) = 0.29786487010104;
    } else if ( tOrder == 4 ) {
      // a_ij( 0, 0 ) = 1.0;
      // a_ij( 1, 0 ) = 0.44437049406734;
      // a_ij( 2, 0 ) = 0.62010185138540;
      // a_ij( 3, 0 ) = 0.17807995410773;
      // a_ij( 4, 0 ) = 0.00683325884039;
      // a_ij( 1, 1 ) = 0.55562950593266;
      // a_ij( 2, 2 ) = 0.37989814861460;
      // a_ij( 4, 2 ) = 0.51723167208978;
      // a_ij( 3, 3 ) = 0.82192004589227;
      // a_ij( 4, 3 ) = 0.12759831133288;
      // a_ij( 4, 4 ) = 0.34833675773694;

      // b_ij( 0, 0 ) = 0.39175222700392;
      // b_ij( 1, 1 ) = 0.36841059262959;
      // b_ij( 2, 2 ) = 0.25189177424738;
      // b_ij( 3, 3 ) = 0.54497475021237;
      // b_ij( 4, 3 ) = 0.08460416338212;
      // b_ij( 4, 4 ) = 0.22600748319395;
      a_ij( 0, 0 ) = 1.0;
      a_ij( 1, 0 ) = 0.444370493651235;
      a_ij( 2, 0 ) = 0.620101851488403;
      a_ij( 3, 0 ) = 0.178079954393132;
      a_ij( 4, 0 ) = 0.000000000000000;
      a_ij( 1, 1 ) = 0.555629506348765;
      a_ij( 2, 2 ) = 0.379898148511597;
      a_ij( 4, 2 ) = 0.517231671970585;
      a_ij( 3, 3 ) = 0.821920045606868;
      a_ij( 4, 3 ) = 0.096059710526147;
      a_ij( 4, 4 ) = 0.386708617503269;

      b_ij( 0, 0 ) = 0.391752226571890;
      b_ij( 1, 1 ) = 0.368410593050371;
      b_ij( 2, 2 ) = 0.251891774271694;
      b_ij( 3, 3 ) = 0.544974750228521;
      b_ij( 4, 3 ) = 0.063692468666290;
      b_ij( 4, 4 ) = 0.226007483236906;
    }
  }
}
