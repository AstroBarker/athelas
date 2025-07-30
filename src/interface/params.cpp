#include <string>

#include "params.hpp"

bool Params::contains(const std::string& key) const {
    return params_.find(key) != params_.end();
}

// Remove a parameter -- note that the caller
// has no guarantee that the key existed.
void Params::remove(const std::string& key) {
    return params_.erase(key);
}

// Clear all parameters
void Params::clear() {
    params_.clear();
}

// Get all parameter keys
std::vector<std::string> keys() const {
    std::vector<std::string> result;
    result.reserve(params_.size());
    for (const auto& [key, _] : params_) {
        result.push_back(key);
    }
    return result;
}
