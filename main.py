import numpy as np
import matplotlib.pyplot as plt
from fortran import streamtracer as f_stream
from numba_code import streamtracer as n_stream
from grid import generate_grid
from plot import plot_streamline_3D, plot_streamline

if __name__ == "__main__":

    # np.random.seed(23849234)

    xc, yc, zc, vx, vy, vz = generate_grid(200)

    # Set Seed Point
    xs0 = np.array([0., 0., 0.])

    # Calculate Streamline

    ns = int(1e6)
    step_size = 0.1

    ds_f = np.zeros(ns)

    xs_f, ns_f = f_stream.calc_streamline(xs0, xc, yc, zc, vx, vy, vz,
                                           ds_f, step_size)
    xs_f = xs_f[:ns_f]
    ds_f = ds_f[:ns_f]
    s_f = ds_f.cumsum()
    
    ds_n = np.zeros(ns)

    xs_n, ns_n = n_stream.calc_streamline(xs0, xc, yc, zc, vx, vy, vz,
                                           ds_n, step_size)
    xs_n = xs_n[:ns_n]
    ds_n = ds_n[:ns_n]
    s_n = ds_n.cumsum()

    # Plot Streamline
    fig = plt.figure(figsize=(8, 8))
    ax = plot_streamline_3D(xs_f)
    plot_streamline_3D(xs_n, ax=ax)

    ax.set(xlim=xc[[0, -1]], ylim=yc[[0, -1]], zlim=zc[[0, -1]])

    fig, ax = plt.subplots()
    plot_streamline(s_f, xs_f, ax=ax)
    ax.set_prop_cycle(None)
    plot_streamline(s_n, xs_n, ax=ax, linestyle='dashed', linewidth=2)

    fig, ax = plt.subplots()
    error = np.sqrt(np.sum((xs_f - xs_n)**2, axis=-1))
    ax.plot(s_n, error)

    plt.show()
