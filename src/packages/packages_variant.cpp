#pragma once
#include <vector>
#include <string_view>
#include <concepts>
#include <span>
#include <variant>
#include <algorithm>
#include <limits>

// Forward declarations for your simulation types
struct SimulationState {
    std::span<double> density;
    std::span<double> velocity;
    std::span<double> energy;
    std::span<double> radiation_energy;
    std::span<double> temperature;
    double time;
    double dt;
    size_t num_cells() const { return density.size(); }
};

struct GridData {
    std::span<const double> cell_centers;
    std::span<const double> cell_sizes;
    std::span<const double> face_areas;
    size_t num_cells;
    double dx_min;  // Minimum cell size for CFL calculations
};

// Physics package concepts
template<typename T>
concept ExplicitPhysicsPackage = requires(T pkg, SimulationState& state, const GridData& grid) {
    { pkg.update_explicit(state, grid) } -> std::same_as<void>;
    { pkg.name() } -> std::convertible_to<std::string_view>;
    { pkg.max_timestep(state, grid) } -> std::convertible_to<double>;
};

template<typename T>
concept ImplicitPhysicsPackage = requires(T pkg, SimulationState& state, const GridData& grid) {
    { pkg.update_implicit(state, grid) } -> std::same_as<void>;
    { pkg.name() } -> std::convertible_to<std::string_view>;
    { pkg.max_timestep(state, grid) } -> std::convertible_to<double>;
};

// =============================================================================
// CONCRETE PHYSICS PACKAGES
// =============================================================================

class HydroPackage {
private:
    double cfl_factor_ = 0.4;
    double gamma_ = 1.4;  // Adiabatic index

public:
    explicit HydroPackage(double cfl = 0.4, double gamma = 1.4) 
        : cfl_factor_(cfl), gamma_(gamma) {}

    void update_explicit(SimulationState& state, const GridData& grid) {
        // Implement your DG hydrodynamics update here
        // This would include:
        // 1. Compute fluxes at cell interfaces using Riemann solvers
        // 2. Apply DG spatial discretization
        // 3. Update conserved quantities (density, momentum, energy)
        
        const size_t n = state.num_cells();
        
        // Example: Simple pressure gradient update (replace with your DG implementation)
        for (size_t i = 1; i < n - 1; ++i) {
            // Compute pressure from EOS
            double pressure_i = (gamma_ - 1.0) * state.energy[i];
            double pressure_l = (gamma_ - 1.0) * state.energy[i-1];
            double pressure_r = (gamma_ - 1.0) * state.energy[i+1];
            
            // Simple centered difference for pressure gradient
            double dp_dx = (pressure_r - pressure_l) / (2.0 * grid.cell_sizes[i]);
            
            // Update momentum (du/dt = -1/rho * dp/dx)
            state.velocity[i] -= state.dt * dp_dx / state.density[i];
            
            // Update energy from kinetic energy change
            state.energy[i] += 0.5 * state.density[i] * 
                              (state.velocity[i] * state.velocity[i]) * state.dt;
        }
    }

    std::string_view name() const { return "Hydrodynamics"; }

    double max_timestep(const SimulationState& state, const GridData& grid) const {
        double dt_min = std::numeric_limits<double>::max();
        
        for (size_t i = 0; i < state.num_cells(); ++i) {
            // Sound speed
            double cs = std::sqrt(gamma_ * (gamma_ - 1.0) * state.energy[i] / state.density[i]);
            // Signal speed (sound + fluid velocity)
            double signal_speed = cs + std::abs(state.velocity[i]);
            // CFL condition
            double dt_cfl = cfl_factor_ * grid.cell_sizes[i] / signal_speed;
            dt_min = std::min(dt_min, dt_cfl);
        }
        
        return dt_min;
    }
};

class RadiationPackage {
private:
    double opacity_;
    double radiation_constant_;
    bool use_implicit_;

public:
    explicit RadiationPackage(double opacity = 1.0, double sigma_rad = 7.56e-15, bool implicit = true)
        : opacity_(opacity), radiation_constant_(sigma_rad), use_implicit_(implicit) {}

    void update_implicit(SimulationState& state, const GridData& grid) {
        // Implement radiation diffusion and coupling
        const size_t n = state.num_cells();
        
        // Example: Simple radiation diffusion (replace with your DG implementation)
        std::vector<double> radiation_old(n);
        for (size_t i = 0; i < n; ++i) {
            radiation_old[i] = state.radiation_energy[i];
        }
        
        // Implicit radiation diffusion solve would go here
        // For now, simple explicit diffusion as placeholder
        for (size_t i = 1; i < n - 1; ++i) {
            double diffusion_coeff = 1.0 / (3.0 * opacity_ * state.density[i]);
            double d2E_dx2 = (radiation_old[i+1] - 2.0*radiation_old[i] + radiation_old[i-1]) /
                            (grid.cell_sizes[i] * grid.cell_sizes[i]);
            
            state.radiation_energy[i] += state.dt * diffusion_coeff * d2E_dx2;
            
            // Coupling term: radiation-matter energy exchange
            double T4 = state.temperature[i] * state.temperature[i] * 
                       state.temperature[i] * state.temperature[i];
            double planck = radiation_constant_ * T4;
            double coupling = opacity_ * state.density[i] * (planck - state.radiation_energy[i]);
            
            state.radiation_energy[i] += state.dt * coupling;
            state.energy[i] -= state.dt * coupling;  // Energy conservation
        }
    }

    std::string_view name() const { return "Radiation Transport"; }

    double max_timestep(const SimulationState& state, const GridData& grid) const {
        if (use_implicit_) {
            return std::numeric_limits<double>::max();  // No explicit timestep limit
        }
        
        // Explicit radiation diffusion timestep limit
        double dt_min = std::numeric_limits<double>::max();
        for (size_t i = 0; i < state.num_cells(); ++i) {
            double diffusion_coeff = 1.0 / (3.0 * opacity_ * state.density[i]);
            double dt_diff = 0.5 * grid.cell_sizes[i] * grid.cell_sizes[i] / diffusion_coeff;
            dt_min = std::min(dt_min, dt_diff);
        }
        return dt_min;
    }
};

class HeatingPackage {
private:
    double heating_rate_;
    std::function<double(double, double)> heating_profile_;  // function of position and time

public:
    explicit HeatingPackage(double rate) : heating_rate_(rate) {
        // Default uniform heating
        heating_profile_ = [rate](double x, double t) { return rate; };
    }

    template<typename Func>
    HeatingPackage(Func&& profile) : heating_rate_(1.0), heating_profile_(std::forward<Func>(profile)) {}

    void update_explicit(SimulationState& state, const GridData& grid) {
        for (size_t i = 0; i < state.num_cells(); ++i) {
            double heating = heating_profile_(grid.cell_centers[i], state.time);
            state.energy[i] += heating * state.dt;
        }
    }

    std::string_view name() const { return "Heating Source"; }

    double max_timestep(const SimulationState& state, const GridData& grid) const {
        // Heating typically doesn't impose strict stability limits
        // But might limit based on energy change rate
        return std::numeric_limits<double>::max();
    }
};

class CoolingPackage {
private:
    double cooling_coefficient_;

public:
    explicit CoolingPackage(double coeff = 1e-12) : cooling_coefficient_(coeff) {}

    void update_explicit(SimulationState& state, const GridData& grid) {
        for (size_t i = 0; i < state.num_cells(); ++i) {
            // Simple cooling proportional to T^2
            double cooling_rate = cooling_coefficient_ * state.density[i] * 
                                 state.temperature[i] * state.temperature[i];
            state.energy[i] -= cooling_rate * state.dt;
            
            // Prevent negative energy
            state.energy[i] = std::max(state.energy[i], 1e-10);
        }
    }

    std::string_view name() const { return "Cooling"; }

    double max_timestep(const SimulationState& state, const GridData& grid) const {
        // Cooling timestep limit to prevent negative temperatures
        double dt_min = std::numeric_limits<double>::max();
        for (size_t i = 0; i < state.num_cells(); ++i) {
            if (state.energy[i] > 0 && state.temperature[i] > 0) {
                double cooling_rate = cooling_coefficient_ * state.density[i] * 
                                     state.temperature[i] * state.temperature[i];
                if (cooling_rate > 0) {
                    double dt_cool = 0.1 * state.energy[i] / cooling_rate;  // 10% energy change limit
                    dt_min = std::min(dt_min, dt_cool);
                }
            }
        }
        return dt_min;
    }
};

// =============================================================================
// VARIANT-BASED PHYSICS MANAGER
// =============================================================================

// Define the variant types
using ExplicitPackageVariant = std::variant<HydroPackage, HeatingPackage, CoolingPackage>;
using ImplicitPackageVariant = std::variant<RadiationPackage>;

class PhysicsManager {
private:
    std::vector<ExplicitPackageVariant> explicit_packages_;
    std::vector<ImplicitPackageVariant> implicit_packages_;

public:
    // Add explicit packages
    template<ExplicitPhysicsPackage T>
    void add_explicit_package(T&& package) {
        explicit_packages_.emplace_back(std::forward<T>(package));
    }

    // Add implicit packages
    template<ImplicitPhysicsPackage T>
    void add_implicit_package(T&& package) {
        implicit_packages_.emplace_back(std::forward<T>(package));
    }

    // Convenience methods for common packages
    void add_hydro(double cfl = 0.4, double gamma = 1.4) {
        add_explicit_package(HydroPackage{cfl, gamma});
    }

    void add_radiation(double opacity = 1.0, bool implicit = true) {
        add_implicit_package(RadiationPackage{opacity, 7.56e-15, implicit});
    }

    void add_heating(double rate) {
        add_explicit_package(HeatingPackage{rate});
    }

    void add_cooling(double coefficient = 1e-12) {
        add_explicit_package(CoolingPackage{coefficient});
    }

    // Update methods - these are your main performance-critical functions
    void update_explicit(SimulationState& state, const GridData& grid) {
        for (auto& pkg_variant : explicit_packages_) {
            std::visit([&](auto& pkg) {
                pkg.update_explicit(state, grid);
            }, pkg_variant);
        }
    }

    void update_implicit(SimulationState& state, const GridData& grid) {
        for (auto& pkg_variant : implicit_packages_) {
            std::visit([&](auto& pkg) {
                pkg.update_implicit(state, grid);
            }, pkg_variant);
        }
    }

    // Timestep calculation
    double min_timestep(const SimulationState& state, const GridData& grid) const {
        double dt_min = std::numeric_limits<double>::max();
        
        for (const auto& pkg_variant : explicit_packages_) {
            std::visit([&](const auto& pkg) {
                dt_min = std::min(dt_min, pkg.max_timestep(state, grid));
            }, pkg_variant);
        }
        
        for (const auto& pkg_variant : implicit_packages_) {
            std::visit([&](const auto& pkg) {
                dt_min = std::min(dt_min, pkg.max_timestep(state, grid));
            }, pkg_variant);
        }
        
        return dt_min;
    }

    // Diagnostic information
    std::vector<std::string_view> get_package_names() const {
        std::vector<std::string_view> names;
        names.reserve(explicit_packages_.size() + implicit_packages_.size());
        
        for (const auto& pkg_variant : explicit_packages_) {
            std::visit([&](const auto& pkg) {
                names.push_back(pkg.name());
            }, pkg_variant);
        }
        
        for (const auto& pkg_variant : implicit_packages_) {
            std::visit([&](const auto& pkg) {
                names.push_back(pkg.name());
            }, pkg_variant);
        }
        
        return names;
    }

    size_t num_explicit_packages() const { return explicit_packages_.size(); }
    size_t num_implicit_packages() const { return implicit_packages_.size(); }
    size_t total_packages() const { return explicit_packages_.size() + implicit_packages_.size(); }
    bool empty() const { return explicit_packages_.empty() && implicit_packages_.empty(); }
};

// =============================================================================
// INTEGRATION WITH RUNGE-KUTTA TIMESTEPPER
// =============================================================================

class RungeKuttaStepper {
private:
    PhysicsManager& physics_;
    GridData grid_;

public:
    explicit RungeKuttaStepper(PhysicsManager& physics, const GridData& grid)
        : physics_(physics), grid_(grid) {}

    // RK4 step with operator splitting for explicit/implicit
    void step_rk4(SimulationState& state) {
        const double dt = state.dt;
        const double dt_half = 0.5 * dt;
        const double dt_sixth = dt / 6.0;
        
        // Store initial state
        SimulationState k1 = state;
        SimulationState k2 = state;
        SimulationState k3 = state;
        SimulationState k4 = state;
        
        // Stage 1
        physics_.update_explicit(k1, grid_);
        physics_.update_implicit(k1, grid_);  // Could be done with different timestep
        
        // Stage 2
        k2.time = state.time + dt_half;
        k2.dt = dt_half;
        // Apply k1 increment scaled by dt/2
        for (size_t i = 0; i < state.num_cells(); ++i) {
            k2.density[i] = state.density[i] + dt_half * (k1.density[i] - state.density[i]) / dt;
            k2.velocity[i] = state.velocity[i] + dt_half * (k1.velocity[i] - state.velocity[i]) / dt;
            k2.energy[i] = state.energy[i] + dt_half * (k1.energy[i] - state.energy[i]) / dt;
            // ... other variables
        }
        physics_.update_explicit(k2, grid_);
        physics_.update_implicit(k2, grid_);
        
        // Stages 3 and 4 would follow similar pattern...
        
        // Final combination (simplified - you'd implement full RK4 combination)
        state = k1;  // This would be the proper RK4 combination
        state.time += dt;
    }

    // Adaptive stepping with timestep control
    void adaptive_step(SimulationState& state, double target_dt) {
        double dt_max = physics_.min_timestep(state, grid_);
        state.dt = std::min(target_dt, dt_max);
        
        step_rk4(state);
    }
};

// =============================================================================
// USAGE EXAMPLES AND FACTORY FUNCTIONS
// =============================================================================

// Standard radiation hydrodynamics setup
inline PhysicsManager create_radhydro_setup() {
    PhysicsManager physics;
    physics.add_hydro(0.4, 1.4);           // CFL=0.4, gamma=1.4
    physics.add_radiation(1.0, true);       // opacity=1.0, implicit=true
    return physics;
}

// Full physics setup with heating and cooling
inline PhysicsManager create_full_setup() {
    PhysicsManager physics;
    physics.add_hydro(0.3, 5.0/3.0);       // Lower CFL, ideal gas
    physics.add_radiation(0.1, true);       // Lower opacity
    physics.add_heating(1e-6);              // Constant heating
    physics.add_cooling(1e-12);             // Cooling
    return physics;
}

// Demonstration of usage
inline void demonstrate_usage() {
    // Create physics setup
    auto physics = create_radhydro_setup();
    
    // Setup simulation state and grid (you'd initialize these properly)
    SimulationState state{};
    GridData grid{};
    
    // Create timestepper
    RungeKuttaStepper stepper(physics, grid);
    
    // Time evolution loop
    double t_final = 1.0;
    double dt_target = 1e-4;
    
    while (state.time < t_final) {
        stepper.adaptive_step(state, dt_target);
        
        // Output, diagnostics, etc.
        if (static_cast<int>(state.time / 0.1) % 10 == 0) {
            auto names = physics.get_package_names();
            // Print diagnostics...
        }
    }
}
//use
:q

