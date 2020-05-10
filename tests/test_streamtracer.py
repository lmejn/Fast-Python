import pytest
import numpy as np
# from numba_code import streamtracer
from fortran_code import streamtracer

# Check bounds

s = [-1, 1]

xs0 = np.array([0., 0., 0.])

step_size = 0.15
ns = int(1e6)

n = 2
x = np.linspace(-1., 1., n)

@pytest.mark.parametrize('roll', range(3))
def test_upper_bound(roll):
    """Test the upper bound end condition.

    Uses a simple velocity field that is constant in one direction. Tests for
    whether the streamline ends outside the bounds.
    """
    v = np.stack([np.ones([n, n, n]),
                  np.zeros([n, n, n]),
                  np.zeros([n, n, n])], axis=0)
    
    ds = np.zeros(ns)

    vxi, vyi, vzi = np.roll(v, roll, axis=0)

    xs, n_steps = streamtracer.calc_streamline(xs0.copy(), x, x, x, vxi, vyi, vzi, ds, step_size)
    xsf = xs[n_steps-1]

    xsf = np.roll(xsf, -roll)

    assert x[1] < xsf[0]
    assert x[0] < xsf[1] < x[1]
    assert x[0] < xsf[2] < x[1]


@pytest.mark.parametrize('roll', range(3))
def test_lower_bound(roll):
    """Test the lower bound end condition.

    Uses a simple velocity field that is constant in one direction. Tests for
    whether the streamline ends outside the bounds.
    """
    v = np.stack([-np.ones([n, n, n]),
                  np.zeros([n, n, n]),
                  np.zeros([n, n, n])], axis=0)

    ds = np.zeros(ns)
    vxi, vyi, vzi = np.roll(v, roll, axis=0)

    xs, n_steps = streamtracer.calc_streamline(xs0.copy(), x, x, x, vxi, vyi, vzi, ds, step_size)
    xsf = xs[n_steps-1]

    xsf = np.roll(xsf, -roll)

    assert xsf[0] < x[0]
    assert x[0] < xsf[1] < x[1]
    assert x[0] < xsf[2] < x[1]