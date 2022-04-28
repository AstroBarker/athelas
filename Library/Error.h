/**
 * File    :  Error.h
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Print error messages ...
 **/

#ifndef ERROR_H
#define ERROR_H

#include <stdexcept>

class Error : public std::runtime_error
{

 public:
  Error( const std::string& message ) : std::runtime_error( message ) {}
};

#endif
