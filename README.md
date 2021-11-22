# splode
(for now) DG Lagrangian hydro

<p align="center">(for now) 1D Lagrangian hydrodynamics solver written in C++ </p>

<p align="center">
<a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>
</p>

* TODO: Everything
 - [x] Start with Lagrange and Legendre polynomial bases
    - [x] We need to put in LG (and LGL) quadratures.
- [x] Make directory for test problem and setup
- [x] Use LAPACKE for lapack calls (more portable)
- [x] Add UpdateGrid()
- [x] Add SlopeLimiter
- [ ] Add TroubledCellIndicator

* Questions
 - [ ] Do I need to be using a modal method?

* BROKEN: 
- [x] Timestepper (nStages > 1)
- [ ] Likely more

# Dependencies
LAPACKE
cBLAS
HDF5
