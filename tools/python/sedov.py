#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from athelas import Athelas
from exactpack.solvers.sedov.sedov import Sedov

plt.style.use("style.mplstyle")


def plot_sedov(chk):
  problem = "sedov"
  fn = f"{problem}_{chk}.h5"
  basis_fn = f"{problem}_basis.h5"

  a = Athelas(fn, basis_fn)
  r = a.r
  print(r[0])
  tau = a.uCF[0, :, 0]
  # vel = a.uCF[1, :, 0]
  # emT = a.uCF[2, :, 0]
  # em = emT - 0.5 * vel * vel
  rho = 1.0 / tau
  # gamma = 1.4
  # p = (gamma - 1.0) * em / tau

  fig, ax = plt.subplots(figsize=(3.5, 3.5))
  plt.minorticks_on()
  #  pre_color = "#94a76f"
  #  vel_color = "#d08c60"
  #  sie_color = "#b07aa1"
  rho_color = "#7095b8"

  # --- analytic solution ---
  t_final = a.time
  solver = Sedov(
    geometry=3,  # spherical
    eblast=0.5,
    gamma=1.4,
    omega=0.0,
  )

  xsol = np.linspace(0.0, 1.0, 64)
  sol = solver._run(xsol, t_final)
  #  ax.plot(xsol, sol["pressure"], color=pre_color, ls=" ", marker="o", alpha=0.75)
  #  ax.plot(xsol, sol["pressure"], color=pre_color, ls=" ", marker="o", alpha=1.0, fillstyle="none")
  ax.plot(xsol, sol["density"], color=rho_color, ls=" ", marker="o", alpha=0.75)
  # ax.plot(xsol, sol["velocity"], color=vel_color, ls=" ", marker="o", alpha=0.75)
  # ax.plot(xsol, sol["pressure"], color=pre_color, ls=" ", marker="o", alpha=0.75)
  ax.plot(
    xsol,
    sol["density"],
    color=rho_color,
    ls=" ",
    marker="o",
    alpha=1.0,
    fillstyle="none",
  )
  # ax.plot(
  #  xsol,
  #  sol["velocity"],
  #  color=vel_color,
  #  ls=" ",
  #  marker="o",
  #  alpha=1.0,
  #  fillstyle="none",
  # )
  # ax.plot(
  #  xsol,
  #  sol["pressure"],
  #  color=pre_color,
  #  ls=" ",
  #  marker="o",
  #  alpha=1.0,
  #  fillstyle="none",
  # )

  # --- athelas ---
  ax.plot(r, rho, label="Density", color=rho_color)
  # ax.plot(r, vel, label="Velocity", color=vel_color)
  # ax.plot(r, p, label="Pressure", color=pre_color)
  #  ax.plot(r, p, label="Pressure", color=pre_color)

  ## limiting
  # for i in range(len(r)):
  #  if a.slope_limiter[i] == 1:
  #    ax.axvline(r[i], color="#7c8c8c", alpha=0.25)

  ax.legend(frameon=False)
  ax.set(ylabel=r"$\rho$ [a.u.]", xlabel="x [a.u.]")

  plt.savefig(f"{problem}_{chk}.png")
  plt.close(fig)


def main():
  chk = "final"
  plot_sedov(chk)


if __name__ == "__main__":
  main()
