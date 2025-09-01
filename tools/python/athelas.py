#!/usr/bin/env python3

from typing import Optional

from basis import ModalBasis

import h5py
import numpy as np


class Athelas:
  """
  Athelas Python class
  """

  def __init__(self, fn, basis_fn=None):
    self.uCF: Optional[np.ndarray] = None
    self.uAF: Optional[np.ndarray] = None
    self.time = None  # type: Optional[float]
    self.sOrder = None  # type: Optional[int]
    self.nX = None  # type: Optional[int]

    self.r = None  # spatial grid
    self.r_nodal = None  # nodal spatial grid
    self.dr = None  # widths

    # indices
    self.idx = {
      "tau": 0,
      "velocity": 1,
      "energy": 2,
      "rad_energy": 3,
      "rad_flux": 4,
    }

    self._load(fn)

    self.basis = None
    if basis_fn is not None:
      self.basis = ModalBasis(basis_fn)

    assert self.uCF is not None
    assert self.uAF is not None

  # End __init__

  def __str__(self):
    return (
      f"--- Athelas Parameters ---\n"
      f"Time: {self.time}\n"
      f"Order: {self.sOrder}\n"
      f"Cells: {self.nX}\n"
      f"Basis loaded: {self.basis is not None}\n"
    )

  # End __str__

  def _load_variable(self, f, name):
    """Load a variable from HDF5 and reshape into (nX, sOrder)"""
    return f[name][:]

  def _load(self, fn):
    """
    load athelas output
    """

    try:
      with h5py.File(fn, "r") as f:
        self.time = f["metadata/time"][0]
        self.sOrder = f["metadata/order"][0]
        self.nX = f["metadata/nx"][0]

        self.r = f["grid/x"][:]
        self.r_nodal = f["grid/x_nodal"][:]
        self.dr = f["grid/dx"][:]

        self.uCF = self._load_variable(f, "variables/conserved")
        self.uAF = self._load_variable(f, "variables/auxiliary")
    except (OSError, KeyError) as e:
      raise RuntimeError(f"Failed to load file '{fn}': {e}")

  #    self.slope_limiter = f["diagnostic/limiter"][:]

  # TODO:

  # End _load

  def get(self, var, k=0):
    assert self.uCF is not None
    idx = self.idx.get(var)
    if idx is None:
      raise KeyError(f"Unknown variable '{var}'")

    return self.uCF[:, k, idx]

  # End get_

  def basis_eval(self, iQ, iX, iEta):
    """
    Evaluate polynomial for quantity iQ at quadrature point iEta on cell iX
    """
    assert self.uCF is not None
    assert self.basis is not None

    result = 0.0
    for k in range(self.basis.order):
      result += self.basis.phi[iX, iEta, k] * self.uCF[iX, k, iQ]
    return result

  def plot(self, ax, var, logx=False, logy=False, label=None, color="#b07aa1"):
    """
    Basic plot function: plot quantity var on axis ax
    """

    q = self.get(var, k=0)
    x = self.r
    if logy:
      q = np.log10(q)
    if logx:
      x = np.log10(x)

    ax.plot(x, q, label=label, color=color)

  # End plot


# End Athelas

if __name__ == "__main__":
  # Examples
  # a = Athelas("sod_final.h5", "sod_basis.h5")
  print("athelas!")
