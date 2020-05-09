"""Test functions for Numba interpolation."""
import pytest
import numpy as np
from interpolate import locate_in_grid, trilinear_interp

# Test grid location

x_grid = np.linspace(-1, 1, 7)
dx = np.diff(x_grid)

x = 0.5*(x_grid[1:] + x_grid[:-1])
x = np.hstack([x[0]-dx[0], x, x[-1]+dx[-1]])

n = len(x_grid)

index = np.hstack([0, np.arange(n-1), n-2])

test_data = [(xi, ix) for xi, ix in zip(x, index)]

@pytest.mark.parametrize('xi, index', test_data)
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

# Test Trilinear Interpolation

def test_trilinear_middle():
    """Test the middle of the cell, which should be the mean of the vertices."""
    fc = np.random.rand(2, 2, 2)
    fi = trilinear_interp(0.5, 0.5, 0.5, fc)

    assert fc.mean() == pytest.approx(fi)

i = [0, 1]
test_data = [[ix, iy, iz] for ix in i for iy in i for iz in i]

@pytest.mark.parametrize('ix, iy, iz', test_data)
def test_trilinear_vertex(ix, iy, iz):
    """Test the vertex value."""
    fc = np.random.rand(2, 2, 2)
    fi = trilinear_interp(float(ix), float(iy), float(iz), fc)

    assert fc[ix, iy, iz] == pytest.approx(fi)
