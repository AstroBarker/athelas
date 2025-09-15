import argparse
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt

from athelas import Athelas

plt.style.use("style.mplstyle")


def parse_args():
  parser = argparse.ArgumentParser(
    description="Plot variable from multiple files."
  )
  parser.add_argument(
    "files", nargs="+", help="List of data files to load and plot"
  )
  parser.add_argument(
    "--logx", action="store_true", help="Use log scale for X axis"
  )
  parser.add_argument(
    "--logy", action="store_true", help="Use log scale for Y axis"
  )
  parser.add_argument(
    "-c", "--color", type=str, default="#b07aa1", help="Color for plotting"
  )
  parser.add_argument(
    "--num-procs",
    type=int,
    default=2,
    help="Maximum number of parallel processes (default: 2)",
  )

  return parser.parse_args()


def plot_file(filename, file_idx, color, logx=False, logy=False):
  data = Athelas(filename)
  print(f"Plotting {filename}: 'rho'")

  # Example dummy data
  x = data.r
  print(x)
  print(np.diff(x))
  y = 1.0 / data.uCF[:, 0, 0]
  print(len(y))

  fig, ax = plt.subplots()
  if logx:
    plt.xscale("log")
  if logy:
    plt.yscale("log")

  plt.minorticks_on()
  ax.plot(x, y, label=filename, color=color, marker="o")
  ax.set(xlabel="x", ylabel=r"$\rho$")
  fn = f"fig_rho_{file_idx:05d}.png"
  print(fn)
  plt.savefig(fn)
  plt.close(fig)


def main():
  args = parse_args()
  with concurrent.futures.ThreadPoolExecutor(
    max_workers=args.num_procs
  ) as executor:
    futures = [
      executor.submit(plot_file, f, i, args.color, args.logx, args.logy)
      for i, f in enumerate(args.files)
    ]
    concurrent.futures.wait(futures)


if __name__ == "__main__":
  main()
