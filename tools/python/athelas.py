#!/usr/bin/env python3

from basis import ModalBasis

import h5py
import matplotlib.pyplot as plt
import numpy as np


class Athelas:
  """
  Athelas Python class
  """

  def __init__(self, fn, basis_fn = None):
    self.uCF = None
    self.time = None
    self.sOrder = None  # spatial order
    self.nX = None  # number of cells

    self.r = None  # spatial grid
    self.dr = None  # widths

    self.load_(fn)
    print(self.time)

    self.basis = None
    if basis_fn is not None:
      self.basis = ModalBasis(basis_fn)

  # End __init__

  def __str__(self):
    print(" --- Athelas Parameters ---")
    return ""

  # End __str__

  def load_(self, fn):
    """
    load athelas output
    """

    with h5py.File(fn, "r") as f:
      self.time = f["Metadata/Time"][0]
      self.sOrder = f["Metadata/Order"][0]
      self.nX = f["Metadata/nX"][0]

      self.r = f["Spatial Grid/Grid"][:]
      self.dr = f["Spatial Grid/Widths"][:]

      nvars = 3
      self.uCF = np.zeros((nvars, self.nX, self.sOrder))

      n = self.nX
      for i in range(self.sOrder):
        self.uCF[0, :, i] = f["/Conserved Fields/Specific Volume"][
          (i * n) : ((i + 1) * n)
        ]
        self.uCF[1, :, i] = f["/Conserved Fields/Velocity"][
          (i * n) : ((i + 1) * n)
        ]
        self.uCF[2, :, i] = f["/Conserved Fields/Specific Internal Energy"][
          (i * n) : ((i + 1) * n)
        ]

      self.slope_limiter = f["Diagnostic Fields/Limiter"][:]

      # TODO:
      # uPF, uAF, uCR

    # End load_

  def basis_eval(self, iF, iX, iEta):
    """
    Evaluate polynomial for quantity iF at quadrature point iEta on cell iX
    """

    assert self.basis is not None

    result = 0.0
    for k in range(self.basis.order):
      result += self.basis.phi[iX, iEta, k] * self.uCF[iF,iX,k]
    return result

  def plot(self, iF, ax, logx = False, logy = False, label = None, color = "teal"):
    """
    Basic plot function: plot quantity index iF from uCF.
    TODO: generalize to select uCF, uCR, etc
    """
      
    q = self.uCF[iF, :, 0]
    x = self.r
    if (logy):
      q = np.log10(q)
    if (logx):
      x = np.log10(x)
    
    ax.plot(x, q, label=label, color = color)
  # End plot


# End Athelas

if __name__ == "__main__":
  # Examples 
  a = Athelas("athelas_Sod_final.h5", "athelas_basis_Sod.h5")
  print(a.basis_eval(0, 10, 2))
  fig, ax = plt.subplots()
  ax.plot( a.r, 1.0 / a.uCF[0,:,0] )
  plt.savefig("test.png")
