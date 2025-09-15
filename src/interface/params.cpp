#include <string>

#include "params.hpp"

auto Params::contains(const std::string &key) const -> bool {
  return params_.contains(key);
}

// Remove a parameter -- note that the caller
// has no guarantee that the key existed.
void Params::remove(const std::string &key) { params_.erase(key); }

// Clear all parameters
void Params::clear() { params_.clear(); }

// Get all parameter keys
auto Params::keys() const -> std::vector<std::string> {
  std::vector<std::string> result;
  result.reserve(params_.size());
  for (const auto &[key, _] : params_) {
    result.push_back(key);
  }
  return result;
}

auto Params::get_type(const std::string &key) const -> std::type_index {
  return params_.at(key)->type();
}
