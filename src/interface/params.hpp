#pragma once

#include <string>
#include <unordered_map>
#include <any>
#include <stdexcept>

class Params {
private:
    std::unordered_map<std::string, std::any> params_;

public:
    Params() = default;

    // Generic setter
    template <typename T>
    void Set(const std::string& key, const T& value) {
        params_[key] = value;
    }

    // Generic getter
    template <typename T>
    T Get(const std::string& key) const {
        auto it = params_.find(key);
        if (it == params_.end()) {
            throw std::runtime_error("Parameter '" + key + "' not found");
        }
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Type mismatch for parameter '" + key + "'");
        }
    }

    // Getter with default value
    template <typename T>
    T Get(const std::string& key, const T& default_value) const {
        auto it = params_.find(key);
        if (it == params_.end()) {
            return default_value;
        }
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            return default_value;
        }
    }

    // Check if a parameter exists
    bool Has(const std::string& key) const;
};
