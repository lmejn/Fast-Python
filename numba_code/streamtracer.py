import numba as nb
import numpy as np
from .interpolate import locate_in_grid, scale_to_grid, trilinear_interp

@nb.njit('float64[:](float64[:],'
                    'float64[:], float64[:], float64[:],'
                    'float64[:,:,:], float64[:,:,:], float64[:,:,:])')
def update(xs, x, y, z, vx, vy, vz):
    """Calculate the update step.

    The update step given by the velocity inputs.

    Args:
        xs (np.array(float)): position of streamline
        x (np.array(float)): x grid
        y (np.array(float)): y grid
        z (np.array(float)): z grid
        vx (np.array(float)): x velocity on grid
        vy (np.array(float)): y velocity on grid
        vz (np.array(float)): z velocity on grid

    Returns:
        np.array(float): update step
    """
    ix = locate_in_grid(xs[0], x)
    iy = locate_in_grid(xs[1], y)
    iz = locate_in_grid(xs[2], z)

    vxc = vx[ix:ix+2, iy:iy+2, iz:iz+2]
    vyc = vy[ix:ix+2, iy:iy+2, iz:iz+2]
    vzc = vz[ix:ix+2, iy:iy+2, iz:iz+2]

    xl = scale_to_grid(xs[0], x[ix], x[ix+1])
    yl = scale_to_grid(xs[1], x[iy], x[iy+1])
    zl = scale_to_grid(xs[2], x[iz], x[iz+1])

    vx = trilinear_interp(xl, yl, zl, vxc)
    vy = trilinear_interp(xl, yl, zl, vyc)
    vz = trilinear_interp(xl, yl, zl, vzc)

    vmag = np.sqrt(vx**2 + vy**2 + vz**2)

    if(vmag==0):
        return np.array([0., 0., 0.])

    return np.array([vx, vy, vz])/vmag

@nb.guvectorize('float64[:],'
                'float64[:], float64[:], float64[:],'
                'float64[:,:,:], float64[:,:,:], float64[:,:,:],'
                'float64[:], float64,'
                'float64[:,:], int64[:]',
                '(nc),'
                '(nx), (ny), (nz),'
                '(nx,ny,nz), (nx,ny,nz), (nx,ny,nz),'
                '(ns),()->(ns,nc), ()'
                )
def calc_streamline(xs0, x, y, z, vx, vy, vz, s, ds, xs, n_steps):
    """Calculate the streamline for a given seed point and velocity grid.

    Args:
        xs0 (np.array(float)): seed point
        x (np.array(float)): x grid positions
        y (np.array(float)): y grid positions
        z (np.array(float)): z grid positions
        vx (np.array(float)): x velocity on grid
        vy (np.array(float)): y velocity on grid
        vz (np.array(float)): z velocity on grid
        s (np.array(float)): array containing number of streamtracer steps
        ds (float): streamtracer step size

    Returns:
        np.array(float): array of streamline positions
    """
    ns = len(s)
    xi = xs0

    for i in range(ns):

        if(xi[0] < x[0] or x[-1] <= xi[0] or
           xi[1] < y[0] or y[-1] <= xi[1] or
           xi[2] < z[0] or z[-1] <= xi[2]):
           break

        k1 = ds*update(xi,          x, y, z, vx, vy, vz)
        k2 = ds*update(xi + 0.5*k1, x, y, z, vx, vy, vz)
        k3 = ds*update(xi + 0.5*k2, x, y, z, vx, vy, vz)
        k4 = ds*update(xi + k3,     x, y, z, vx, vy, vz)

        xi += (k1 + 2*(k2 + k3) + k4)/6.

        xs[i] = xi
        s[i] = (i+1)*ds
    
    n_steps[0] = i
