#pragma once
/**
 * @file history.hpp
 * --------------
 *
 * @brief HistoryOutput class:
 */

#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry/grid.hpp"
#include "polynomial_basis.hpp"
#include "state/state.hpp"

class HistoryOutput {
 public:
  // NOTE: We are always passing in two basis objects to our history functions.
  // This hsould be fine -- the driver always has two, even in pure Hydro mode.
  using QuantityFunction =
      std::function<double(const State &, const GridStructure &,
                           const ModalBasis *, const ModalBasis *)>;

  explicit HistoryOutput(const std::string &filename, bool enabled);

  void add_quantity(const std::string &name, QuantityFunction func);

  void write(const State &state, const GridStructure &grid,
             const ModalBasis *fluid_basis, const ModalBasis *rad_basis,
             double time);

 private:
  bool enabled_;
  bool header_written_;
  std::string filename_;
  std::ofstream file_;
  std::unordered_map<std::string, QuantityFunction> quantities_;
  std::vector<std::string> quantity_names_;
};
