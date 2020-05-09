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


@nb.njit('float64(float64, float64, float64, float64[:,:,:])')
def trilinear_interp(xl, yl, zl, fc):
    """Tri-Linear interpolation on a given cell.

    Takes in the scaled position within the cell [0 to 1] and cell vertex
    values and returns the interpolated value. Values outside [0 to 1] are
    extrapolated.

    Args:
        xl (float): scaled x position [0 to 1]
        yl (float): scaled y position [0 to 1]
        zl (float): scaled z position [0 to 1]
        fc (np.array(float)): cell values as the vertex

    Returns:
        float: interpolated value
    """
    m_xl = 1 - xl
    m_yl = 1 - yl

    c00 = fc[0, 0, 0]*m_xl + fc[1, 0, 0]*xl
    c01 = fc[0, 0, 1]*m_xl + fc[1, 0, 1]*xl
    c10 = fc[0, 1, 0]*m_xl + fc[1, 1, 0]*xl
    c11 = fc[0, 1, 1]*m_xl + fc[1, 1, 1]*xl

    c0 = c00*m_yl + c10*yl
    c1 = c01*m_yl + c11*yl

    return c0*(1 - zl) + c1*zl
