import numpy as np
import matplotlib.pyplot as plt
import numba_code.streamtracer as nb_streamtracer
from grid import generate_grid

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

    # Plot Streamline

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.plot(xs[:, 0], xs[:, 1], xs[:, 2])
    ax.plot(xs[[0], 0], xs[[0], 1], xs[[0], 2], '.', color=p[0].get_color())
    ax.plot(xs[[-1], 0], xs[[-1], 1], xs[[-1], 2], 'x', color=p[0].get_color())
    ax.set(xlim=xc[[0, -1]], ylim=yc[[0, -1]], zlim=zc[[0, -1]],
           xlabel='x', ylabel='y', zlabel='z')
    plt.show()
