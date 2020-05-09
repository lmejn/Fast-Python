import numba as nb

@nb.njit('int64(float64, float64[:])')
def locate_in_grid(xi, x_grid):
    """Locates the grid index containing the point xi.

    Locates the index by bisecting the grid. If xi is outside the grid, it
    returns the start/end indices. 

    Args:
        xi (float): position to be located in grid
        x_grid (np.array(float)): position of grid points

    Returns:
        int: index of grid where xi is between
    """
    n = len(x_grid)
    if(xi < x_grid[0]):
        return 0
    if(x_grid[-1] <= xi):
        return n-2

    i0 = 0
    i1 = n-2

    for _ in range(n):
        im = (i0 + i1)//2

        if(x_grid[im] <= xi and xi < x_grid[im+1]):
            return im

        if(xi < x_grid[im]):
            i1 = im
        elif(x_grid[im+1] < xi):
            i0 = im+1

    return 0