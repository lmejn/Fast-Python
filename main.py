import numpy as np
import matplotlib.pyplot as plt
import numba_code.streamtracer as nb_streamtracer


def poly3D(x, y, z, coeff):
    """Sample a polynomial 3D scalar field

    Polynomial order is specified by the number of elements in each row in
    coeff. Number of rows should always ==3. e.g. polynomial of order 2
    requires coeff of shape [3, 2].

    Args:
        x (float): x position
        y (float): y position
        z (float): z position
        coeff (np.array): Coefficients of the polynomial scalar field
    
    Returns:
        value of scalar field
    """       
    fx = np.polyval(coeff[0], x)
    fy = np.polyval(coeff[1], y)
    fz = np.polyval(coeff[2], z)

    return fx*fy*fz


def create_grid_axis(n, lim):
    """Create a uniformly spaced axis

    Args:
        n (int): number of cells
        lim (list): list containing lower and upper bound
    
    Returns:
        xb: np.array containing cell boundary positions
        xc: np.array containing cell centre positions
        dx: np.array containing cell sizes
    """
    xb = np.linspace(lim[0], lim[1], n+1)
    xc = 0.5*(xb[1:]+xb[:-1])
    dx = np.diff(xb)

    return xb, xc, dx


if __name__ == "__main__":

    # Generate grid
    n = np.array([1, 1, 1])*200

    xb, xc, dx = create_grid_axis(n[0], [-1, 1])
    yb, yc, dy = create_grid_axis(n[1], [-1, 1])
    zb, zc, dz = create_grid_axis(n[2], [-1, 1])

    Yc, Xc, Zc = np.meshgrid(yc, xc, zc)

    # Store polynomial coefficients
    cx = 2*np.random.rand(3, 3)-1
    cy = 2*np.random.rand(3, 3)-1
    cz = 2*np.random.rand(3, 3)-1

    # Calculate vector field
    vx = poly3D(Xc, Yc, Zc, cx)
    vy = poly3D(Xc, Yc, Zc, cy)
    vz = poly3D(Xc, Yc, Zc, cz)

    # Set Seed Point
    xs0 = np.array([0., 0., 0.])

    # Calculate Streamline

    ns = int(1e6)
    ds = dx[0]/10.
    s = np.zeros(ns)

    xs, n_steps = nb_streamtracer.calc_streamline(xs0, xc, yc, zc, vx, vy, vz,
                                                  s, ds)
    xs = xs[:n_steps]

    # Plot Streamline

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.plot(xs[:, 0], xs[:, 1], xs[:, 2])
    ax.plot(xs[[0], 0], xs[[0], 1], xs[[0], 2], '.', color=p[0].get_color())
    ax.plot(xs[[-1], 0], xs[[-1], 1], xs[[-1], 2], 'x', color=p[0].get_color())
    ax.set(xlim=xb[[0, -1]], ylim=yb[[0, -1]], zlim=zb[[0, -1]],
           xlabel='x', ylabel='y', zlabel='z')
    plt.show()
