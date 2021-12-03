# splode
(for now) DG Lagrangian hydro

<p align="center">(for now) 1D Lagrangian hydrodynamics solver written in C++ </p>

<p align="center">
<a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>
</p>

* TODO: Transitioning to modal basis
 - [x] Write Taylor functions
 - [x] Write functions to orthogonalize them
 - [ ] Output overhaul - write basis terms and all coefficients
 - [x] Separate out nNodes from order where they should be distinct.
    - Broken
 - [x] We need to replace instances of Lagrange with Taylor, etc
    - [x] SlopeLimiter needs changes, FluidDiscretization... 
    - [x] Initialization?
    - [x] Node Coordinate? Grid?
 - [ ] Fix BoudnaryConditions

* TODO: Everything
 - [ ] We need a build system....
 - [x] Update Grid to depend opn GridStructures
 - [x] Start with Lagrange and Legendre polynomial bases
    - [x] We need to put in LG (and LGL) quadratures.
- [x] Make directory for test problem and setup
- [x] Use LAPACKE for lapack calls (more portable)
- [x] Add UpdateGrid()
- [x] Add SlopeLimiter
- [ ] Add TroubledCellIndicator
- [ ] 4th order timestepper

* Future Work

- We will want to extend beyond the minmod limiter to something which allows us to retain high order information.
- Parallelism


* BROKEN: 
- [ ] nNodes > order, crash when SlopeLimiter applied.
- [x] Timestepper (nStages > 1)
- [ ] Likely more

# Dependencies
LAPACKE
cBLAS
HDF5
