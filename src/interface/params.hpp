#pragma once
/**
 * @file params.hpp
 * --------------
 *
 * @brief Simple params object
 * @note Syntactic sugar around
 */

#include <string>
#include <unordered_map>
#include <any>
#include <stdexcept>

#include "utils/error.hpp"

class Params {
private:
    std::unordered_map<std::string, std::any> params_;

public:
    Params() = default;

    template <typename T>
    void add(const std::string& key, const T& value) {
        params_[key] = value;
    }

    template <typename T>
    T get(const std::string& key) const {
        auto it = params_.find(key);
        if (it == params_.end()) {
            THROW_ATHELAS_ERROR("Parameter '" + key + "' not found");
        }
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            THROW_ATHELAS_ERROR("Type mismatch for parameter '" + key + "'");
        }
    }

    template <typename T>
    T get_or_add(const std::string& key, const T& default_value) {
        auto it = params_.find(key);
        if (it == params_.end()) {
            add(key, default_value);
            return default_value;
        }
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            THROW_ATHELAS_ERROR("Type mismatch for parameter '" + key + "'");
        }
    }

    // Check if a parameter exists
    bool contains(const std::string& key) const;
};
