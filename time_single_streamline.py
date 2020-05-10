import numpy as np
import matplotlib.pyplot as plt
from fortran_code import streamtracer as f_stream
from numba_code import streamtracer as n_stream
from grid import generate_grid
import timeit



if __name__ == "__main__":

    xc, yc, zc, vx, vy, vz = generate_grid(200, name='poly')

    # Set Seed Point
    xs0 = np.array([0., 0., 0.])

    # Calculate Streamline

    ns = int(1e6)
    step_size = 0.1


    # Fortran
    ds_f = np.zeros(ns)
    def fortran_calc():
        _, _ = f_stream.calc_streamline(xs0, xc, yc, zc, vx, vy, vz,
                                        ds_f, step_size)

    result_fortran = timeit.timeit("fortran_calc()",
                                   setup="from __main__ import fortran_calc",
                                   number=100)
    print(f"Fortran: {result_fortran}")

    # Numba
    ds_n = np.zeros(ns)
    def numba_calc():
        _, _ = n_stream.calc_streamline(xs0, xc, yc, zc, vx, vy, vz,
                                        ds_n, step_size)

    result_numba = timeit.timeit("numba_calc()",
                                   setup="from __main__ import numba_calc",
                                   number=100)
    print(f"Numba: {result_numba}")
