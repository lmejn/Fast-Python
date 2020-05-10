import pytest
import numpy as np
from numba_code.streamtracer import calc_streamline

# Check bounds

s = [-1, 1]

xs0 = np.array([0., 0., 0.])
x = np.array([-1., 1.])

step_size = 0.1
ds = np.zeros(int(100))

@pytest.mark.parametrize('roll', range(3))
def test_upper_bound(roll):
    """Test the upper bound end condition.

    Uses a simple velocity field that is constant in one direction. Tests for
    whether the streamline ends outside the bounds.
    """
    v = np.stack([np.ones([2, 2, 2]),
                  np.zeros([2, 2, 2]),
                  np.zeros([2, 2, 2])], axis=0)

    vxi, vyi, vzi = np.roll(v, roll, axis=0)

    xs, n_steps = calc_streamline(xs0.copy(), x, x, x, vxi, vyi, vzi, ds, step_size)
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
    v = np.stack([-np.ones([2, 2, 2]),
                  np.zeros([2, 2, 2]),
                  np.zeros([2, 2, 2])], axis=0)

    vxi, vyi, vzi = np.roll(v, roll, axis=0)

    xs, n_steps = calc_streamline(xs0.copy(), x, x, x, vxi, vyi, vzi, ds, step_size)
    xsf = xs[n_steps-1]

    xsf = np.roll(xsf, -roll)

    assert xsf[0] < x[0]
    assert x[0] < xsf[1] < x[1]
    assert x[0] < xsf[2] < x[1]