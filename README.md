# athelas (A radiaTion-Hydrodynamics codE for modeLing supernovA Systems (WIP) )
<p align="center">sypherically symmetric Lagrangian radiation-hydrodynamics solver written in C++ </p>

<p align="center">
<a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>
</p>

Currently, `athelas` solves the 1D Cartesian Lagrangian equation of non-relativistic hydrodynamics using a discontinuous Galerkin scheme. 
It will be extended to spherical symmetry, special relativistic hydrodynamics.
For now, it includes an ideal gas equation of state.

Future work will implement a finite element Poisson solver for Gravity and multiground flux-limited diffusion for radiation.

# Installation:
`athelas` is installed using cmake. From the root directory of `athelas`, run the following:

```sh
mkdir build && cd build
cmake ..
cmake --build .
```

This will create a directory `bin` in the root directory that contains the executable.

## NOTE: 
The build system may not be perfect yet. I have hard coded in paths to my libraries for HDF5, LAPACK, and BLAS as cmake had some issues finding them. You may overwrite those lines, or comment them out and try to let `find_package()` do the work.

# TODO: Spherical Symetry
 - [ ] Ensure mass conservation...
## Issues with mass conservation 
 - weirdness in r_cm calculation. eta_cm is zero. can I calculate r_cm from eta_cm? 

# TODO: Transitioning to modal basis
 - [x] 4th order timestepper
 - [ ] Output overhaul - write basis terms and all coefficients
   - partially done
 - [x] Write Taylor functions
 - [x] Write functions to orthogonalize them

 - [x] Separate out nNodes from order where they should be distinct.
    - Broken
 - [x] We need to replace instances of Lagrange with Taylor, etc
    - [x] SlopeLimiter needs changes, FluidDiscretization... 
    - [x] Initialization?
    - [x] Node Coordinate? Grid?
 - [x] Fix BoundaryConditions

# TODO:
 - [ ] Initialize with input file at runtime
 - [ ] Create DataStructure1D?
 - [x] **We need a build system....**
 - [x] TimeStepper class (main purpose: hold U_s, SumVar, etc)
 - [x] Update Grid to depend opn GridStructures
 - [x] Start with Lagrange and Legendre polynomial bases
    - [x] We need to put in LG (and LGL) quadratures.
- [x] Make directory for test problem and setup
- [x] Use LAPACKE for lapack calls (more portable)
- [x] Add UpdateGrid()
- [x] Add SlopeLimiter
- [x] Add TroubledCellIndicator

## Reader
 - [ ] Need to extend Reader to compute solution at arbitrary points using basis
    - Have to output the full solution and basis first.

# Future Work

- We will want to extend beyond the minmod limiter to something which allows us to retain high order information.
- Relativistic hydro
- Parallelism
- Poisson solver
- Multigroup flux-limited diffusion


# BUGS: 
- [ ] nNodes > order, crash when SlopeLimiter applied.

# Dependencies
* LAPACKE
* cBLAS
* HDF5
