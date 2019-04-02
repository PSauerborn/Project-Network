from scipy.optimize import curve_fit, least_squares
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.optimize import minimize
from scipy.optimize import fmin_cg
from pyprind import ProgBar
import matplotlib.cm as cm
import threading
# from gutzwiller_functions import *
from multiprocessing import Process
from mpl_toolkits.mplot3d import Axes3D


style.use(['bmh', r'C:\directory_python\research\bose_hubbard\Gutzwiller\extensions\threaded_application\cythonVersion\settings.mplstyle'])


def plot_data(server_count, threads, nrows, ncols, mu=1, N=5, z=6, iters=50, inhomogenous=False, V=0, U=None, uncertainty=False):
    """Function used to assemble the quadrants of parameter space to then plot the resulting phase diagram

    Parameters
    ----------
    nrows: int
        number of rows in parameter space
    ncols: int
        number of columns in parameter space
    mu: float
        chemical potential
    z: int
        coordination number
    iters: int
        number of iterations in each quadrant

    """

    # the individual quadrants must first be reassembled in the correct order

    frame = np.empty((nrows*server_count, ncols*server_count), dtype=object)

    for set in threads:
        data = set.data
        print(data)

        for array, location in data:
            row, col = location
            frame[row, col] = array

    data = None

    for i in range(nrows*server_count):
        row = None
        for j in range(ncols):


            try:
                row = np.concatenate((row, frame[i, j]), axis=1) if row is not None else frame[i,j]
            except Exception as err:
                print(err)
                print(i, j, frame[i, j])

        if data is None:
            data = row
        else:
            data = np.concatenate((data, row), axis=0)

    data = data[::-1,:]

    import seaborn as sns

    fig, ax = plt.subplots()

    plot = ax.imshow(data,
                    interpolation='spline16', cmap=cm.gist_heat)

    cbar = plt.colorbar(plot)
    cbar.set_label(r'$|\psi|$', rotation='horizontal',
                    fontsize=20, labelpad=20)

    arrow = ax.arrow(y=2.2, x=1, dx=2, dy=3)

    ax.set_xlabel(r'$\frac{wz}{U}$', fontsize=35, labelpad=30)
    ax.set_ylabel(r'$\frac{\mu}{U}$', fontsize=35, rotation='horizontal', labelpad=30)

    # ax.set_title(r'$Phase \ Diagram \ for \ U \ = \ 1 \ at \ n_{max} \ = \ 6$', fontsize=30, pad=30)

    ax.set_xticks(np.linspace(0, 1000, 5))
    ax.set_yticks(np.linspace(0, 1000, 4))

    ticks = np.linspace(0, 0.45, 5)
    # ticks = [0, 0.05, 0.1, 0.15, 0.2]

    ax.set_xticklabels(ticks, rotation=0)
    ax.set_yticklabels([i for i in reversed(range(4))], rotation=0)

    fig.tight_layout()

    return fig




# nrows, ncols, nquads, iters = solve_system(N_sites=9, N=5, z=6, mu=1, nrows=4, ncols=4, V=0., target=plot_n)

# plot_data(nrows=5, ncols=5, iters=80, N=5, mu=1, seaborn=False)
