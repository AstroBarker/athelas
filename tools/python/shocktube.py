#!/usr/bin/env python3

import glob
import sys

from astropy import constants as consts
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

from athelas import Athelas
from exactpack.solvers.riemann.ep_riemann import IGEOS_Solver, streakplot

plt.style.use("style.mplstyle")


def plot_shocktube(chk):
  problem = "sod"
  fn = f"{problem}_{chk}.h5"
  basis_fn = f"{problem}_basis.h5"

  a = Athelas(fn, basis_fn)
  r = a.r
  tau = a.uCF[0, :, 0]
  vel = a.uCF[1, :, 0]
  emT = a.uCF[2, :, 0]
  em = emT - 0.5 * vel * vel
  rho = 1.0 / tau
  gamma = 1.4
  p = (gamma - 1.0) * em / tau

  fig, ax = plt.subplots(figsize=(3.5, 3.5))
  plt.minorticks_on()
  rho_color = "#94a76f"
  vel_color = "#d08c60"
  sie_color = "#b07aa1"
  pre_color = "#7095b8"

  # --- analytic solution ---
  t_final = a.time
  solver = IGEOS_Solver(
    rl=1.0,
    ul=0.0,
    pl=1.0,
    gl=1.4,
    rr=0.125,
    ur=0.0,
    pr=0.1,
    gr=1.4,
    xmin=0.0,
    xd0=0.5,
    xmax=1.0,
    t=t_final,
  )

  xsol = np.linspace(0.0, 1.0, 32)
  sol = solver._run(xsol, t_final)
  # streakplot(solver=solver, soln=sol, xs=xsol, t=t_final, N=101, var_str="pressure")
  ax.plot(xsol, sol["pressure"], color=pre_color, ls=" ", marker="o")
  ax.plot(xsol, sol["density"], color=rho_color, ls=" ", marker="o")
  ax.plot(xsol, sol["velocity"], color=vel_color, ls=" ", marker="o")
  ax.plot(
    xsol,
    sol["specific_internal_energy"] / 2.5,
    color=sie_color,
    ls=" ",
    marker="o",
  )

  # --- athelas ---
  ax.plot(r, rho, label="Density", color=rho_color)
  ax.plot(r, vel, label="Velocity", color=vel_color)
  ax.plot(r, em / 2.5, label="Energy / 2.5", color=sie_color)
  ax.plot(r, p, label="Pressure", color=pre_color)

  # limiting
  for i in range(len(r)):
    if a.slope_limiter[i] == 1:
      ax.axvline(r[i], color="#7c8c8c", alpha=0.25)

  ax.legend(frameon=False)
  ax.set(ylabel=r"Solution", xlabel="x")

  plt.savefig(f"sod_{chk}.png")


def main():
  chk = "final"
  plot_shocktube(chk)


if __name__ == "__main__":
  main()
