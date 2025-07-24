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
    self.uCR: Optional[np.ndarray] = None
    self.time = None
    self.sOrder = None  # spatial order
    self.nX = None  # number of cells

    self.r = None  # spatial grid
    self.dr = None  # widths

    # indices
    self.idx = {
      "tau": 0,
      "velocity": 1,
      "energy": 2,
      "rad_energy": 0,
      "rad_flux": 1,
    }

    self._load(fn)

    self.basis = None
    if basis_fn is not None:
      self.basis = ModalBasis(basis_fn)

    assert self.uCF is not None
    assert self.uCR is not None

  # End __init__

  def __str__(self):
    return (
      f"--- Athelas Parameters ---\n"
      f"Time: {self.time}\n"
      f"Order: {self.sOrder}\n"
      f"Cells: {self.nX}\n"
      f"Grid range: {self.r[0]} - {self.r[-1]} (Δr ~ {np.mean(self.dr)})\n"
      f"Basis loaded: {self.basis is not None}\n"
    )

  # End __str__

  def _load_variable(self, f, name, shape):
    """Load a variable from HDF5 and reshape into (nX, sOrder)"""
    flat = f[f"{name}"][:]
    return flat.reshape((self.sOrder, self.nX)).T  # shape: (nX, sOrder)

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
        self.dr = f["grid/dx"][:]

        nvars = 3
        self.uCF = np.zeros((nvars, self.nX, self.sOrder))
        self.uCR = np.zeros((2, self.nX, self.sOrder))
        assert self.uCF is not None
        assert self.uCR is not None

        self.uCF[0] = self._load_variable(
          f, "conserved/tau", (self.nX, self.sOrder)
        )
        self.uCF[1] = self._load_variable(
          f, "conserved/velocity", (self.nX, self.sOrder)
        )
        self.uCF[2] = self._load_variable(
          f, "conserved/energy", (self.nX, self.sOrder)
        )
        try:
          self.uCR[0] = self._load_variable(
            f, "conserved/rad_energy", (self.nX, self.sOrder)
          )
          self.uCR[1] = self._load_variable(
            f, "conserved/rad_momentum", (self.nX, self.sOrder)
          )
        except KeyError as e:
          print(f"KeyError: {e}")
          self.uCR = np.zeros_like(self.uCR[0])
    except (OSError, KeyError) as e:
      raise RuntimeError(f"Failed to load file '{fn}': {e}")

#    self.slope_limiter = f["diagnostic/limiter"][:]

    # TODO:
    # uPF, uAF, uCR

    # End _load

  def get(self, var, k=0):
    assert self.uCF is not None
    assert self.uCR is not None
    idx = self.idx.get(var)
    if idx is None:
      raise KeyError(f"Unknown variable '{var}'")

    return self.uCR[idx, :, k] if "rad" in var else self.uCF[idx, :, k]

  # End get_

  def basis_eval(self, iQ, iX, iEta):
    """
    Evaluate polynomial for quantity iQ at quadrature point iEta on cell iX
    """
    assert self.uCF is not None
    assert self.uCR is not None
    assert self.basis is not None

    result = 0.0
    for k in range(self.basis.order):
      result += self.basis.phi[iX, iEta, k] * self.uCF[iQ, iX, k]
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
