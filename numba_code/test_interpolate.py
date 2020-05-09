"""Test functions for Numba interpolation."""
import pytest
import numpy as np
from interpolate import locate_in_grid


x_grid = np.linspace(-1, 1, 5)
dx = np.diff(x_grid)

x = 0.5*(x_grid[1:] + x_grid[:-1])
x = np.hstack([x[0]-dx[0], x, x[-1]+dx[-1]])

n = len(x_grid)

index = np.hstack([0, np.arange(n-2), n-2])

data = [(xi, ix) for xi, ix in zip(x, index)]

@pytest.mark.parametrize('xi, index', data)
def test_grid_indices(xi, index):
    """Test whether locate_in_grid returns the correct index."""
    ix = locate_in_grid(xi, x_grid)
    assert ix == index


@pytest.mark.parametrize('x', x[1:-1])
def test_point_in_cell(x):
    """Test whether the position is consistent with the cell index."""
    ix = locate_in_grid(x, x_grid)
    assert x_grid[ix] <= x < x_grid[ix+1]


@pytest.mark.parametrize('x', x[[0, -1]])
def test_outside_points(x):
    """Test points outside the grid, which should not return an error."""
    ix = locate_in_grid(x, x_grid)
    x_grid[ix], x_grid[ix+1]
