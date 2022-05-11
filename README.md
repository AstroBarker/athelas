# athelas (A radiaTion-Hydrodynamics codE for modeLing supernovA Systems (WIP) )
<p align="center">sypherically symmetric Lagrangian radiation-hydrodynamics solver written in C++ </p>

<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>
</p>

Currently, `athelas` solves the 1D Lagrangian equation of non-relativistic hydrodynamics using a discontinuous Galerkin scheme. It includes planar geometry and spherical symmetry.
It will be extended to special relativistic hydrodynamics.
For now, it includes an ideal gas equation of state.

Future work will implement gravity and multigroup two moment radiation.


# In Progress: Kokkos
We use Kokkos for parallelism. 
Currently, most significant data structures use `Kokkos::Views` and loops are parallelised with Kokkos.
More work to port other parts of the code.
You need to install Kokkos (instructions will be included in time).

# Installation:
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


# BUGS: 
- [ ] nNodes > order, crash when SlopeLimiter applied.
- [x] Issue with TCI
- [x] Issues with Characteristic Limiting

# Dependencies
* Kokkos
* LAPACKE
* cBLAS
* HDF5


## Clang-format

We use clang format for code cleanliness. 
The current version of `clang-format` used is `clang-format-13`.
Simply call `Tools/Bash/format.sh` to format the `.h` and `.cpp` files.


# TODO: Spherical Symetry
 - [x] Ensure mass conservation...
 - Done?


# TODO: Transitioning to modal basis
 - [ ] Output overhaul - write basis terms and all coefficients

# TODO:
 - [ ] Initialize with input file at runtime

## Reader
 - [ ] Need to extend Reader to compute solution at arbitrary points using basis
    - Have to output the full solution and basis first.