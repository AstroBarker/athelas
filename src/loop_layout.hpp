#pragma once

// Inspiration from [athelas](https://github.com/athelas-hpc-lab/athelas)

// Default loop patterns for kokkos parallel loop wrappers
// see kokkos_abstraction.hpp for available tags.
// Kokkos tight loop layout
#define DEFAULT_LOOP_PATTERN athelas::loop_pattern_mdrange_tag
#define DEFAULT_FLAT_LOOP_PATTERN athelas::loop_pattern_flatrange_tag

// Kokkos hierarchical loop layout
#define DEFAULT_OUTER_LOOP_PATTERN athelas::outer_loop_pattern_teams_tag
#define DEFAULT_INNER_LOOP_PATTERN athelas::inner_loop_pattern_simdfor_tag
