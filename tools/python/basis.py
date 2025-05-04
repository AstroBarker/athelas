#!/usr/bin/env python3

import h5py


class ModalBasis:
  """
  Athelas modal basis class
  """

  def __init__(self, fn):
    self.order = None  # spatial order
    self.phi = None

    self.load_(fn)

  # End __init__

  def __str__(self):
    print(" --- Basis Parameters ---")
    print(f" ~ Order : {self.order}")
    return ""

  # End __str__

  def load_(self, fn):
    """
    load athelas output
    """

    with h5py.File(fn, "r") as f:
      self.phi = f["basis"][:, :, :]  # nX, order + 2, order

    self.order = len(self.phi[0, 0, :])

  # End load_


# End ModalBasis
