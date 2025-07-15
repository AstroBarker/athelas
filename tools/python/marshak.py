#!/usr/bin/env python3

import glob
import sys

from astropy import constants as consts
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

from athelas import Athelas
from exactpack.solvers.suolson.timmes import suolson  # import timmes
from exactpack.solvers.suolson import SuOlson  # import timmes

plt.style.use("style.mplstyle")


def plot_marshak(chk):
  problem = "marshak"
  fn = f"{problem}_{chk}.h5"
  basis_fn = f"{problem}_basis.h5"

  c = consts.c.cgs.value
  sigma = consts.sigma_sb.cgs.value
  a_rad = 4.0 * sigma / c
  epsilon = 1.0
  alpha_so = 4.0 * a_rad / epsilon

  a = Athelas(fn, basis_fn)
  r = a.r
  tau = a.uCF[0, :, 0]
  rho = 1.0 / tau
  vel = a.uCF[1, :, 0]
  emT = a.uCF[2, :, 0]
  em = emT - 0.5 * vel * vel
  ev = em * rho
  gamma = 1.4
  p = (gamma - 1.0) * em / tau

  # use Su-Olson form for temperature
  T_g = np.power(4.0 * ev / alpha_so, 0.25)

  # rad
  ev_r = a.uCR[0, :, 0]
  T_r = np.power(ev_r / a_rad, 0.25)

  fig, ax = plt.subplots(figsize=(3.5, 3.5))
  plt.minorticks_on()
  pre_color = "#94a76f"
  vel_color = "#d08c60"
  sie_color = "#b07aa1"
  rho_color = "#7095b8"

  # --- analytic solution ---
  EV_TO_K = 1.0 / consts.k_B.to("eV / K").value
  print("EV TO K", EV_TO_K)
  t_final = a.time
  rmin = a.r[0]
  rmax = 3.466205e-3
  print(rmax)
  print(a.r[-1])
  r_sol = np.linspace(rmin, rmax, 1024)
  kappa = 577.0
  solver = SuOlson()
  t_bndry = 3.481334e6  # K
  #soln = solver(r_sol, t_final)
  # rad_sol, fluid_sol = suolson(t_final, r_sol, t_bndry, kappa, alpha_so)
  # rad_sol *= EV_TO_K
  # fluid_sol *= EV_TO_K
  # ax.plot(r_sol, rad_sol, ls = " ", marker="o", color=pre_color)
  # ax.plot(r_sol, fluid_sol, ls = " ", marker="o", color=vel_color)
  #r_sol = soln["position"]
  x_sol, t_rad_sol, t_fluid_sol = np.loadtxt("marshak.dat", unpack=True, usecols=(1,4,5))
  A = x_sol[0]
  B = x_sol[-1]
  x_sol = ((x_sol - A) * (rmax - rmin)/(B - A)) + rmin
  print(x_sol)
  print(r)
  ax.semilogx(
    x_sol,
    t_rad_sol * t_bndry,
    ls=" ",
    marker="o",
    color=pre_color,
    alpha=0.75,
  )
  ax.semilogx(
    x_sol,
    t_fluid_sol * t_bndry,
    ls=" ",
    marker="o",
    color=vel_color,
    alpha=0.75,
  )

  ax.semilogx(
    x_sol,
    t_rad_sol * t_bndry,
    ls=" ",
    marker="o",
    color=pre_color,
    alpha=1.0,
    fillstyle="none",
  )
  ax.semilogx(
    x_sol,
    t_fluid_sol * t_bndry,
    ls=" ",
    marker="o",
    color=vel_color,
    alpha=1.0,
    fillstyle="none",
  )

  # --- athelas ---
  ax.semilogx(r, T_g, label="Fluid", color=vel_color)
  ax.semilogx(r, T_r, label="Radiation", color=pre_color, ls="--")
  #  ax.plot(r, p, label="Pressure", color=pre_color)

  ## limiting
  # for i in range(len(r)):
  #  if a.slope_limiter[i] == 1:
  #    ax.axvline(r[i], color="#7c8c8c", alpha=0.25)

  ax.legend(frameon=False)
  ax.set(ylabel=r"Temperature [K]", xlabel="x [cm]")

  svname = f"{problem}_{chk}.png"
  print(f"Saving figure {svname}")
  plt.savefig(svname)
  plt.close(fig)


def main():
  chk = "final"
  plot_marshak(chk)


if __name__ == "__main__":
  main()
