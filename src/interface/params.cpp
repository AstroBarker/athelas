#include <string>

#include "params.hpp"

bool Params::contains(const std::string& key) const {
    return params_.find(key) != params_.end();
}
