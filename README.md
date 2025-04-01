# athelas (A radiaTion-Hydrodynamics codE for modeLing sphericAl Supernovae )

<p align="center">sypherically symmetric Lagrangian radiation-hydrodynamics solver written in C++ </p>

[![Build](https://github.com/AstroBarker/athelas/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/AstroBarker/athelas/actions/workflows/cmake-multi-platform.yml)
<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>
</p>

Currently, `athelas` solves the 1D Lagrangian equation of non-relativistic hydrodynamics using a discontinuous Galerkin scheme. It includes planar geometry and spherical symmetry.
It will be extended to special relativistic hydrodynamics.

# Kokkos
We use [Kokkos](https://github.com/kokkos) for shared memory parallelism. 
Currently, most significant data structures use `Kokkos::Views` and loops are parallelised with `Kokkos`.
More work to port other parts of the code.

# TOML++
We use [toml++](https://github.com/marzer/tomlplusplus) for parsing input files in the `.toml` format.

# Installation
`athelas` uses submodules to include dependencies. 
The best way to get the source is the following 
```sh
git clone --recursive git@github.com:AstroBarker/athelas.git
```

# Building
`athelas` is built using `cmake`. From the root directory of `athelas`, run the following:

```sh
mkdir build && cd build
cmake ..
cmake --build .
```

As a temporary fix for Ubuntu CI, we need to pass a `MACHINE` flag.
On Mac we support `-DMACHINE=MACOS`,
Ubuntu supports `-DMACHINE=UBUNTU` (primarily because the CI fails to find `lapacke.h` unless we hold its hand and this is how we do that, for now.)
This places the executable in the `build` dir.


## NOTE: 
The build system may not be perfect yet. Your mileage may vary.

# Code Style

We use `clang format` and `ruff` for code cleanliness. 
Rules are listed in `.clang-format`.
The current version of `clang-format` used is 20.1.0.
Simply call `tools/bash/format.sh` to format the `.hpp` and `.cpp` files.

Python code linting and formatting is done with `ruff`. 
Rules are listed in `ruff.toml`. 
To check all python in the current directory, you may `ruff ..`
To format a given file according to `ruff.toml`, run `ruff format file.py`. 

Checks for formatting are performed on each PR.

# Dependencies
* LAPACKE
* HDF5
* Kokkos (avialable as a submodule)
* TOML++ (submodule)

Hopefully `lapacke` won't be necessary forever, but at present it is needed for initializing the quadrature.
I find that, on Arch Linux systems, `lapack`, `lapacke`, and `openblas` is sufficient for all `lapacke` needs.
