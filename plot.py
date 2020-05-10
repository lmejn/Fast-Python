"""Streamline Plotting Functions."""
import numpy as np
import matplotlib.pyplot as plt

def plot_streamline_3D(xs, ax=None, end_points=True, **kwargs):
    """Plot a streamline in 3D.

    Args:
        xs (np.array(float)): positions of streamline
        ax (plt.axes, optional): axes to plot on. Defaults to None.
        end_points (bool, optional): plot end points. Defaults to True.

    Kwargs:
        keyword arguments for plt.plot

    Returns:
        [plt.axes]: generated or supplied axes object
    """
    if(ax is None):
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')

    p = ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], **kwargs)
    if(end_points):
        ax.plot(xs[[0], 0], xs[[0], 1], xs[[0], 2], '.',
                color=p[0].get_color())
        ax.plot(xs[[-1], 0], xs[[-1], 1], xs[[-1], 2], 'x',
                color=p[0].get_color())
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    return ax

def plot_streamline(s, xs, ax=None, **kwargs):
    """Plot the position of the streamline against streamline length.

    Args:
        s (np.array(float)): length along streamline
        xs (np.array(float)): positions of streamline
        ax (plt.axes, optional): axes to plot on. Defaults to None.

    Kwargs:
        keyword arguments for plt.plot

    Returns:
        [plt.axes]: generated or supplied axes object
    """    
    if(ax is None):
        ax = plt.gca()
    
    ax.plot(s, xs, **kwargs)
    ax.legend('xyz')

    return ax