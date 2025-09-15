#!/usr/bin/env python3

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from athelas import Athelas
from exactpack.solvers.riemann.ep_riemann import IGEOS_Solver  # , streakplot

plt.style.use("style.mplstyle")


def plot_shocktube(chk):
  problem = "sod"
  fn = f"{problem}_{chk}.h5"
  basis_fn = f"{problem}_basis.h5"

  a = Athelas(fn, basis_fn)
  r = a.r
  tau = a.uCF[:, 0, 0]
  vel = a.uCF[:, 0, 1]
  emT = a.uCF[:, 0, 2]
  em = emT - 0.5 * vel * vel
  rho = 1.0 / tau
  gamma = 1.4
  p = (gamma - 1.0) * em / tau

  fig, ax = plt.subplots(figsize=(3.5, 3.5))
  plt.minorticks_on()
  rho_color = "#86e3a1"  # green
  vel_color = "#ff9a8b"  # orange
  sie_color = "#d287ef"  # purple
  pre_color = "#8cc8f3"  # blue

  # --- analytic solution ---
  t_final = a.time
  solver = IGEOS_Solver(
    rl=1.0,
    ul=-2.0,
    pl=0.4,
    gl=gamma,
    rr=1.0,
    ur=2.0,
    pr=0.4,
    gr=gamma,
    xmin=0.0,
    xd0=0.5,
    xmax=1.0,
    t=t_final,
  )

  xsol = np.linspace(0.0, 1.0, 48)
  sol = solver._run(xsol, t_final)
  # streakplot(solver=solver, soln=sol, xs=xsol, t=t_final, N=101, var_str="pressure")
  plt.scatter(
    xsol,
    sol["density"],
    s=18,
    facecolor=mcolors.to_rgba(rho_color, alpha=0.25),
    edgecolor=mcolors.to_rgba(rho_color, alpha=1.0),
    linewidth=0.5,
    label="Analytic Solution",
  )
  plt.scatter(
    xsol,
    sol["pressure"],
    s=18,
    facecolor=mcolors.to_rgba(pre_color, alpha=0.25),
    edgecolor=mcolors.to_rgba(pre_color, alpha=1.0),
    linewidth=0.5,
  )
  plt.scatter(
    xsol,
    sol["specific_internal_energy"],
    s=18,
    facecolor=mcolors.to_rgba(sie_color, alpha=0.25),
    edgecolor=mcolors.to_rgba(sie_color, alpha=1.0),
    linewidth=0.5,
  )

  ax.plot(r, rho, label="Density", color=rho_color)
  ax.plot(r, em, label="Specific Internal Energy", color=sie_color)
  ax.plot(r, p, label="Pressure", color=pre_color)

  # limiting
  #  for i in range(len(r)):
  #    if a.slope_limiter[i] == 1:
  #      ax.axvline(r[i], color="#7c8c8c", alpha=0.25)

  ax.legend(frameon=False, fontsize=6)
  ax.set(ylabel=r"Solution", xlabel="x", xlim=[0.0, 1.0])

  plt.savefig(f"einfeldt_{chk}.png")


def main():
  chk = "final"
  plot_shocktube(chk)
  return os.EX_OK


if __name__ == "__main__":
  main()
