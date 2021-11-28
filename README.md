# splode
(for now) DG Lagrangian hydro

<p align="center">(for now) 1D Lagrangian hydrodynamics solver written in C++ </p>

<p align="center">
<a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>
</p>

* TODO: Transitioning to modal basis
 - [x] Write Taylor functions
 - [x] Write functions to orthogonalize them
 - [ ] We need to replace instances of Lagrange with Taylor, etc
    - [ ] SlopeLimiter needs changes, FluidDiscretization... 
    - [x] Initialization?
    - [ ] Node Coordinate? Grid?

* TODO: Everything
 - [ ] Update Grid to depend opn GridStructures
 - [x] Start with Lagrange and Legendre polynomial bases
    - [x] We need to put in LG (and LGL) quadratures.
- [x] Make directory for test problem and setup
- [x] Use LAPACKE for lapack calls (more portable)
- [x] Add UpdateGrid()
- [x] Add SlopeLimiter
- [ ] Add TroubledCellIndicator


* BROKEN: 
- [x] Timestepper (nStages > 1)
- [ ] Likely more

# Dependencies
LAPACKE
cBLAS
HDF5
