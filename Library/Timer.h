#ifndef TIMER_H
#define TIMER_H

/**
 * File     :  Timer.h
 * --------------
 *
 * Author   : Brandon L. Barker
 *   Based off of https://gist.github.com/mcleary/b0bf4fa88830ff7c882d
 * Purpose  : Simple timer structure
 **/

#include <chrono>

class Timer
{
 public:
  void start( );
  void stop( );

  double elapsedMilliseconds( );

  double elapsedSeconds( );

 private:
  std::chrono::time_point<std::chrono::steady_clock> m_StartTime;
  std::chrono::time_point<std::chrono::steady_clock> m_EndTime;
  bool m_bRunning = false;
};

#endif