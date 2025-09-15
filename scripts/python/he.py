import argparse

import numpy as np
import matplotlib.pyplot as plt

from athelas import Athelas

plt.style.use("style.mplstyle")


def plot_file(var, filename, file_idx, color, logx=False, logy=False):
  data = Athelas(filename)
  print(f"Plotting {filename}: '{var}'")

  # Example dummy data
  x = data.r
  y = data.get(var, k=0)

  fig, ax = plt.subplots()
  if logx:
    plt.xscale("log")
  if logy:
    plt.yscale("log")

  plt.minorticks_on()
  ax.plot(x, y, label=filename, color=color)
  ax.set(xlabel="x", ylabel=var)
  fig.savefig(f"fig_{var}_{file_idx:05d}.png")
  plt.close(fig)


def main():
  fn_init = "hydrostatic_balance_00000.h5"
  fn_final = "hydrostatic_balance_final.h5"

  data_init = Athelas(fn_init)
  data_final = Athelas(fn_final)

  fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
  pressure_init = data_init.uAF[:, :, 0]
  pressure_final = data_final.uAF[:, :, 0]
  diff = np.abs(pressure_init - pressure_final) / pressure_init

  ax1.loglog(
    data_final.r_nodal,
    pressure_final,
    color="#5891d6",
    label="Final",
    ls=" ",
    marker="o",
  )
  ax1.loglog(data_init.r_nodal, pressure_init, color="k", label="Initial")
  ax1.legend()
  ax1.set(ylabel="P")

  one = np.all(np.diff(pressure_init) <= 0.0)
  two = np.all(np.diff(pressure_final) <= 0.0)
  print(np.where(np.diff(pressure_init) > 0.0))

  print(pressure_init)
  ax2.semilogx(data_init.r_nodal, diff)
  ax2.set(xlabel="r", ylabel="E")
  plt.savefig("he.png")
  plt.close()


if __name__ == "__main__":
  main()
