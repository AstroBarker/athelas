#pragma once

#include <algorithm>
#include <limits>
#include <string_view>
#include <vector>

#include "concepts/packages.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "timestepper.hpp"

// Package wrapper that erases types while maintaining performance
class PackageWrapper {
 public:
  template <PhysicsPackage T>
  explicit PackageWrapper(T&& package)
      : package_(std::make_unique<PackageModel<std::decay_t<T>>>(
            std::forward<T>(package))) {}

  void update_explicit(View3D<double> state, View3D<double> dU,
                       const GridStructure& grid, const TimeStepInfo& dt_info) {
    if (package_->has_explicit()) {
      package_->update_explicit(state, dU, grid, dt_info);
    }
  }

  void update_implicit(View3D<double> state, View3D<double> dU,
                       const GridStructure& grid, const TimeStepInfo& dt_info) {
    if (package_->has_implicit()) {
      package_->update_implicit(state, dU, grid, dt_info);
    }
  }

  [[nodiscard]] auto min_timestep(View3D<double> state, const GridStructure& grid,
                                  const TimeStepInfo& dt_info) const -> double {
    if (package_->is_active()) {
      return package_->min_timestep(state, grid, dt_info);
    }
    return std::numeric_limits<double>::max();
  }

  [[nodiscard]] auto name() const noexcept -> std::string_view {
    return package_->name();
  }
  [[nodiscard]] auto is_active() const noexcept -> bool {
    return package_->is_active();
  }
  [[nodiscard]] auto has_explicit() const noexcept -> bool {
    return package_->has_explicit();
  }
  [[nodiscard]] auto has_implicit() const noexcept -> bool {
    return package_->has_implicit();
  }

 private:
  struct PackageConcept {
    virtual ~PackageConcept()                   = default;
    virtual void update_explicit(View3D<double>, View3D<double>, const GridStructure&,
                                 const TimeStepInfo&) = 0;
    virtual void update_implicit(View3D<double>, View3D<double>, const GridStructure&,
                                 const TimeStepInfo&) = 0;
    [[nodiscard]] virtual auto min_timestep(View3D<double> state,
                                            const GridStructure& grid,
                                            const TimeStepInfo& dt_info) const -> double = 0;

    [[nodiscard]] virtual auto name() const noexcept -> std::string_view = 0;
    [[nodiscard]] virtual auto is_active() const noexcept -> bool        = 0;
    [[nodiscard]] virtual auto has_explicit() const noexcept -> bool     = 0;
    [[nodiscard]] virtual auto has_implicit() const noexcept -> bool     = 0;
  };

  template <PhysicsPackage T>
  struct PackageModel final : PackageConcept {
    explicit PackageModel(T package) : package_(std::move(package)) {}

    void update_explicit(View3D<double> state, View3D<double> dU,
                         const GridStructure& grid, const TimeStepInfo& dt_info) override {
      if constexpr (has_explicit_update_v<T>) {
        package_.update_explicit(state, dU, grid, dt_info);
      }
    }

    void update_implicit(View3D<double> state, View3D<double> dU,
                         const GridStructure& grid, const TimeStepInfo& dt_info) override {
      if constexpr (has_implicit_update_v<T>) {
        package_.update_implicit(state, dU, grid, dt_info);
      }
    }

    [[nodiscard]] auto min_timestep(View3D<double> state, const GridStructure& grid,
                                    const TimeStepInfo& dt_info) const -> double override {
      return package_.min_timestep(state, grid, dt_info);
    }

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
      return package_.name();
    }
    [[nodiscard]] auto is_active() const noexcept -> bool override {
      return package_.is_active();
    }
    [[nodiscard]] auto has_explicit() const noexcept -> bool override {
      return has_explicit_update_v<T>;
    }
    [[nodiscard]] auto has_implicit() const noexcept -> bool override {
      return has_implicit_update_v<T>;
    }

   private:
    T package_;
  };

  std::unique_ptr<PackageConcept> package_;
};

// High-performance package manager with separate explicit/implicit lists
class PackageManager {
 public:
  template <PhysicsPackage T>
  void add_package(T&& package) {
  std::cout << "this = " << static_cast<void*>(this) << "\n";
  std::cout << "explicit_packages_ address = " << &explicit_packages_ << "\n";
  std::cout << "explicit_packages_ size = " << explicit_packages_.size() << "\n";
    auto wrapper = std::make_unique<PackageWrapper>(std::forward<T>(package));

    // TODO(astrobarker): emplace back
    explicit_packages_ = {};
    if (wrapper->has_explicit()) {
      std::println("explicit size {}", explicit_packages_.size());
      explicit_packages_.push_back(wrapper.get());
    }
    if (wrapper->has_implicit()) {
      implicit_packages_.push_back(wrapper.get());
    }
    if (wrapper->has_implicit() && wrapper->has_explicit()) {
      imex_packages_.push_back(wrapper.get());
    }

    all_packages_.push_back(std::move(wrapper));
  }

  void update_explicit(View3D<double> state, View3D<double> dU,
                       const GridStructure& grid, const TimeStepInfo& dt_info) {
    for (auto* pkg : explicit_packages_) {
      if (pkg->is_active()) {
        pkg->update_explicit(state, dU, grid, dt_info);
      }
    }
  }

  void update_implicit(View3D<double> state, View3D<double> dU,
                       const GridStructure& grid, const TimeStepInfo& dt_info) {
    for (auto* pkg : implicit_packages_) {
      if (pkg->is_active()) {
        pkg->update_implicit(state, dU, grid, dt_info);
      }
    }
  }

  auto min_timestep(View3D<double> state, const GridStructure& grid,
                    const TimeStepInfo& dt_info) const -> double {
    double min_dt_info = std::numeric_limits<double>::max();
    // TODO(astrobarker): loop over all_packages_
    // TODO(astrobarker): CFL
    for (auto* pkg : explicit_packages_) {
      if (pkg->is_active()) {
        min_dt_info = std::min(min_dt_info, pkg->min_timestep(state, grid, dt_info));
      }
    }
    for (auto* pkg : implicit_packages_) {
      if (pkg->is_active()) {
        min_dt_info = std::min(min_dt_info, pkg->min_timestep(state, grid, dt_info));
      }
    }
    return min_dt_info;
  }

  void clear() {
    all_packages_.clear();
    explicit_packages_.clear();
    implicit_packages_.clear();
    imex_packages_.clear();
  }

  [[nodiscard]] auto get_package_names() const noexcept
      -> std::vector<std::string_view> {
    std::vector<std::string_view> names;
    names.reserve(all_packages_.size());
    for (const auto& pkg : all_packages_) {
      names.push_back(pkg->name());
    }
    return names;
  }

 private:
  std::vector<std::unique_ptr<PackageWrapper>> all_packages_;
  std::vector<PackageWrapper*> explicit_packages_;
  std::vector<PackageWrapper*> implicit_packages_;
  std::vector<PackageWrapper*> imex_packages_;
};
