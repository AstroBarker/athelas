/**
 * File     :  Timer.h
 * --------------
 *
 * Author   : Brandon L. Barker
 *   Based off of https://gist.github.com/mcleary/b0bf4fa88830ff7c882d
 * Purpose  : Simple timer structure
 * Not currently used, but may be extended.
 **/

#include "Timer.h"

void Timer::start( )
{
  m_StartTime = std::chrono::steady_clock::now( );
  m_bRunning  = true;
}

void Timer::stop( )
{
  m_EndTime  = std::chrono::steady_clock::now( );
  m_bRunning = false;
}

double Timer::elapsedMilliseconds( )
{
  std::chrono::time_point<std::chrono::steady_clock> endTime;

  if ( m_bRunning )
  {
    endTime = std::chrono::steady_clock::now( );
  }
  else
  {
    endTime = m_EndTime;
  }

  return std::chrono::duration_cast<std::chrono::milliseconds>( endTime -
                                                                m_StartTime )
      .count( );
}

double Timer::elapsedSeconds( ) { return elapsedMilliseconds( ) / 1000.0; }