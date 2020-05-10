"""Functions for generating the grid variables."""
import numpy as np

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
    """Create a uniformly spaced axis.

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

def generate_grid(n, name):
    """Generate the grid spacings and velocity variables.

    Args:
        n (int): number of cells in each direction

    Returns:
        xc (np.array): cell centred x position of grid cells
        yc (np.array): cell centred y position of grid cells
        zc (np.array): cell centred z position of grid cells
        vx (np.array): cell centred x velocity of grid cells
        vy (np.array): cell centred y velocity of grid cells
        vz (np.array): cell centred z velocity of grid cells
    """    
    # Generate grid
    n = np.array([n, n, n], dtype=int)

    _, xc, _ = create_grid_axis(n[0], [-1, 1])
    _, yc, _ = create_grid_axis(n[1], [-1, 1])
    _, zc, _ = create_grid_axis(n[2], [-1, 1])

    Yc, Xc, Zc = np.meshgrid(yc, xc, zc)

    # Store polynomial coefficients
    if(name == 'random'):
        cx, cy, cz = create_random_poly_coeff()
    else:
        cx, cy, cz = load_poly_coeff(name)


    # Calculate vector field
    vx = poly3D(Xc, Yc, Zc, cx)
    vy = poly3D(Xc, Yc, Zc, cy)
    vz = poly3D(Xc, Yc, Zc, cz)

    return xc, yc, zc, vx, vy, vz


def create_random_poly_coeff():
    """Create random polynomial coefficients for velocity field.

    Returns:
        np.array: arrays containing the coefficients for vx, vy and vz
    """
    cx = 2*np.random.rand(3, 3)-1
    cy = 2*np.random.rand(3, 3)-1
    cz = 2*np.random.rand(3, 3)-1
    return cx, cy, cz


def save_poly_coeff(name, cx, cy, cz):
    """Save polynomial coefficients for velocity field to file.

    Args:
        name (str): name of coefficients
        cx (np.array): vx coefficients
        cy (np.array): vy coefficients
        cz (np.array): vz coefficients
    """    
    np.save(name + "_cx.npy", cx)
    np.save(name + "_cy.npy", cy)
    np.save(name + "_cz.npy", cz)

def load_poly_coeff(name):
    """Load polynomial coefficients for velocity field from file.

    Args:
        name (str): name of coefficients

    Returns:
        np.array: arrays containing the coefficients for vx, vy and vz
    """
    cx = np.load(name + "_cx.npy")
    cy = np.load(name + "_cy.npy")
    cz = np.load(name + "_cz.npy")

    return cx, cy, cz