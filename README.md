# athelas (A radiaTion-Hydrodynamics codE for modeLing sphericAl Supernovae )

<p align="center">sypherically symmetric Lagrangian radiation-hydrodynamics solver written in C++ </p>

[![Build](https://github.com/AstroBarker/athelas/actions/workflows/cmake.yml/badge.svg)](https://github.com/AstroBarker/athelas/actions/workflows/cmake.yml)
<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>
</p>

Currently, `athelas` solves the 1D Lagrangian equation of non-relativistic hydrodynamics using a discontinuous Galerkin scheme. It includes planar geometry and spherical symmetry.
It will be extended to special relativistic hydrodynamics.
For now, it includes an ideal gas equation of state.

Future work will implement gravity and multigroup two moment radiation.

# Kokkos
We use [Kokkos](https://github.com/kokkos) for shared memory parallelism. 
Currently, most significant data structures use `Kokkos::Views` and loops are parallelised with `Kokkos`.
More work to port other parts of the code.

# SimpleIni
We use [SimpleIni](https://github.com/brofield/simpleini) for parsing input files in the for parsing input files in the `.ini` format.

# Installation
`athelas` uses submodules to include `Kokkos` as a dependency. 
The best way to get the source is the following 
```sh
git clone --recursive git@github.com:AstroBarker/athelas.git
```

# Building
`athelas` is installed using cmake. From the root directory of `athelas`, run the following:

```sh
mkdir build && cd build
cmake ..
cmake --build .
```

This will create a directory `bin` in the root directory that contains the executable.

## NOTE: 
The build system may not be perfect yet. Your mileage may vary.


# Future Work

- Grey M1 radiation
- Relativistic hydro
- Gravity
- Multigroup radiation

# Dependencies
* LAPACKE
* HDF5
* Kokkos (asavilable as a submodule)


## Clang-format

We use clang format for code cleanliness. 
The current version of `clang-format` used is `clang-format-13`.
Simply call `Tools/Bash/format.sh` to format the `.h` and `.cpp` files.

# TODO:
 - [ ] Initialize with input file at runtime
 - [x] Bound enforcing limiter
 - [ ] New TCI
 - [ ] linter
 - [ ] format on make


# BUGS: 
- [ ] nNodes > order, crash when SlopeLimiter applied.
