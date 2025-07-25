#!/usr/bin/env python3

from astropy import constants as consts
import matplotlib.pyplot as plt
import numpy as np

from athelas import Athelas

plt.style.use("style.mplstyle")


# needs a massive cleanup
def plot_marshak(chk):
  problem = "marshak"
  fn = f"{problem}_{chk}.h5"
  basis_fn = f"{problem}_basis.h5"

  kappa = 577
  rho0 = 10.0
  chi = rho0 * kappa

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

  # use Su-Olson form for temperature
  T_g = np.power(4.0 * ev / alpha_so, 0.25)

  # rad
  ev_r = a.uCR[0, :, 0]
  T_r = np.power(ev_r / a_rad, 0.25)

  fig, ax = plt.subplots(figsize=(3.5, 3.5))
  plt.minorticks_on()
  pre_color = "#94a76f"
  vel_color = "#d08c60"

  # --- analytic solution ---
  kappa = 577.0
  t_bndry = 3.481334e6  # K
  x_sol, t_rad_sol, t_fluid_sol = np.loadtxt(
    "marshak.dat", unpack=True, usecols=(1, 4, 5)
  )
  x_sol = np.sqrt(3) * x_sol / chi
  ax.semilogx(
    x_sol,
    t_rad_sol * t_bndry,
    ls=" ",
    marker="o",
    color=pre_color,
    alpha=0.5,
    markersize=2.0,
  )
  ax.semilogx(
    x_sol,
    t_fluid_sol * t_bndry,
    ls=" ",
    marker="o",
    color=vel_color,
    alpha=0.5,
    markersize=2.0,
  )

  ax.semilogx(
    x_sol,
    t_rad_sol * t_bndry,
    ls=" ",
    marker="o",
    color=pre_color,
    alpha=0.75,
    fillstyle="none",
    markersize=2.0,
  )
  ax.semilogx(
    x_sol,
    t_fluid_sol * t_bndry,
    ls=" ",
    marker="o",
    color=vel_color,
    alpha=0.75,
    fillstyle="none",
    markersize=2.0,
  )

  # --- athelas ---
  #  try to buid x vars
  new_r = np.zeros_like(r)
  for i in range(len(r)):
    new_r[i] = np.sqrt(3) * r[i]
  ax.semilogx(new_r, T_g, label="Fluid", color=vel_color)
  ax.semilogx(new_r, T_r, label="Radiation", color=pre_color, ls="--")

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
