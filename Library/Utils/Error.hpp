/**
 * File    :  Error.hpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Print error messages ...
 **/

#ifndef _ERROR_HPP_
#define _ERROR_HPP_

#include <stdexcept>
#include <assert.h>     /* assert */

class Error : public std::runtime_error
{

 public:
  Error( const std::string &message ) : std::runtime_error( message ) {}
};

#endif // _ERROR_HPP_
