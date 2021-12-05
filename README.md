# athelas (A radiaTion-Hydrodynamics codE for modeLing supernovA Systems (WIP)
<p align="center">sypherically symmetric Lagrangian radiation-hydrodynamics solver written in C++ </p>

<p align="center">
<a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>
</p>

athelas solves the 1D Cartesian Lagrangian equation of non-relativistic hydrodynamics using a discontinuous Galerkin scheme. 
It will be extended to spherical symmetry, special relativistic hydrodynamics.
For now, it includes an ideal gas equation of state.

Future work will a finite element Poisson solver for Gravity and multiground flux-limited diffusion for radiation.

* TODO: Transitioning to modal basis
 - [ ] 4th order timestepper
 - [ ] Output overhaul - write basis terms and all coefficients
 - [x] Write Taylor functions
 - [x] Write functions to orthogonalize them

 - [x] Separate out nNodes from order where they should be distinct.
    - Broken
 - [x] We need to replace instances of Lagrange with Taylor, etc
    - [x] SlopeLimiter needs changes, FluidDiscretization... 
    - [x] Initialization?
    - [x] Node Coordinate? Grid?
 - [ ] Fix BoundaryConditions

* TODO:
 - [ ] **We need a build system....**
 - [x] TimeStepper class (main purpose: hold U_s, SumVar, etc)
 - [ ] Initialize with input file at runtime
 - [x] Update Grid to depend opn GridStructures
 - [x] Start with Lagrange and Legendre polynomial bases
    - [x] We need to put in LG (and LGL) quadratures.
- [x] Make directory for test problem and setup
- [x] Use LAPACKE for lapack calls (more portable)
- [x] Add UpdateGrid()
- [x] Add SlopeLimiter
- [x] Add TroubledCellIndicator

* Reader
 - [ ] Need to extend Reader to compute solution at arbitrary points using basis
    - Have to output the full solution and basis first.

* Future Work

- We will want to extend beyond the minmod limiter to something which allows us to retain high order information.
- Parallelism
- Poisson solver
- Multigroup flux-limited diffusion


* BUGS: 
- [ ] nNodes > order, crash when SlopeLimiter applied.

# Dependencies
* LAPACKE
* cBLAS
* HDF5
