import numpy as np
import matplotlib.pyplot as plt
import numba_code.streamtracer as nb_streamtracer
from grid import generate_grid
from plot import plot_streamline_3D, plot_streamline

if __name__ == "__main__":

    xc, yc, zc, vx, vy, vz = generate_grid(200)

    # Set Seed Point
    xs0 = np.array([0., 0., 0.])

    # Calculate Streamline

    ns = int(1e4)
    step_size = 0.1
    ds = np.zeros(ns)

    xs, n_steps = nb_streamtracer.calc_streamline(xs0,
                                                  xc, yc, zc, vx, vy, vz,
                                                  ds, step_size)
    xs = xs[:n_steps]
    ds = ds[:n_steps]
    s = ds.cumsum()

    # Plot Streamline
    fig = plt.figure(figsize=(8, 8))
    ax = plot_streamline_3D(xs)
    ax.set(xlim=xc[[0, -1]], ylim=yc[[0, -1]], zlim=zc[[0, -1]])

    fig, ax = plt.subplots()
    ax = plot_streamline(s, xs, ax=ax)

    plt.show()
