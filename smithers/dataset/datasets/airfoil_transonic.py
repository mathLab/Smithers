from ..abstract_dataset import AbstractDataset
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import CubicSpline


class AirfoilTransonicDataset(AbstractDataset):

    parametric = True
    time_dependent = False
    description = "Transonic flow past a NACA 0012 airfoil"
    data_directory = os.path.join(os.path.dirname(__file__),
            'airfoil_transonic')

    def __init__(self):
        '''
        Initialize the dataset and structure it in dictionaries.
        '''
        self.params = np.load(os.path.join(self.data_directory, 'params.npy'))
        v_internal = np.load(os.path.join(self.data_directory,
            'snapshots_v_internal.npy'))
        p_internal = np.load(os.path.join(self.data_directory,
            'snapshots_p_internal.npy'))
        p_airfoil = np.load(os.path.join(self.data_directory,
            'snapshots_p_airfoil.npy'))
        wn_airfoil = np.load(os.path.join(self.data_directory,
            'snapshots_wn_airfoil.npy'))

        self.snapshots = {
                'internal': {'mag(v)': v_internal, 'p': p_internal},
                'airfoil': {'p': p_airfoil, 'wallshear_normal': wn_airfoil}
        }
        coords_internal = np.load(os.path.join(self.data_directory,
            'coords_internal.npy'))
        coords_airfoil = np.load(os.path.join(self.data_directory,
            'coords_airfoil.npy'))
        self.pts_coordinates = {
                'internal': coords_internal, 'airfoil': coords_airfoil
                }
        self.auxiliary_triang = tri.Triangulation(coords_internal[0, :],
                coords_internal[1, :])
        self.auxiliary_triang.set_mask(self._mask_airfoil())

    def _coords_airfoil(self, which='pos'):
        '''
        Get the positive (if which='pos') or negative (if which='neg')
        coordinates of the airfoil.
        '''
        x_airfoil = self.pts_coordinates['airfoil'][0, :]
        y_airfoil = self.pts_coordinates['airfoil'][1, :]
        if which=='pos':
            indices = y_airfoil >= 0
        elif which=='neg':
            indices = y_airfoil <= 0

        x_airfoil_which = sorted(x_airfoil[indices])
        y_airfoil_which = y_airfoil[indices][
                np.argsort(x_airfoil[indices])]
        return x_airfoil_which, y_airfoil_which

    def _f_airfoil_neg(self, tol=1e-4):
        '''
        Return a function that represents the negative part of the airfoil.
        '''
        x_airfoil_neg, y_airfoil_neg = self._coords_airfoil(which='neg')
        f_neg = CubicSpline(x_airfoil_neg, y_airfoil_neg)
        array = []
        for x in self.pts_coordinates['internal'][0, :]:
            if x >= 0 and x <= 1:
                array.append(f_neg(x) - tol)
            else:
                array.append(np.nan)
        return np.array(array)

    def _f_airfoil_pos(self, tol=1e-4):
        '''
        Return a function that represents the positive part of the airfoil.
        '''
        x_airfoil_pos, y_airfoil_pos = self._coords_airfoil(which='pos')
        f_pos = CubicSpline(x_airfoil_pos, y_airfoil_pos)
        array = []
        for x in self.pts_coordinates['internal'][0, :]:
            if x >= 0 and x <= 1:
                array.append(f_pos(x) + tol)
            else:
                array.append(np.nan)
        return np.array(array)

    def _mask_airfoil(self):
        '''
        Mask the triangles that are inside the airfoil (useful for plots).
        '''
        coords = self.pts_coordinates['internal']
        x_airfoil = self.pts_coordinates['airfoil'][0, :]
        y_airfoil = self.pts_coordinates['airfoil'][1, :]
        isbad = (np.greater(coords[0, :], x_airfoil.min()) &
                np.less(coords[0, :], x_airfoil.max()) &
                np.less(coords[1, :], self._f_airfoil_pos()) &
                np.greater(coords[1, :], self._f_airfoil_neg()))
        triangles = self.auxiliary_triang.triangles
        mask = np.all(np.where(isbad[triangles], True, False), axis=1)
        return mask

    def plot_internal(self, snaps_array, idx=0, title='Snapshot', namefig=None,
            lim_x=(-0.5, 2), lim_y=(-0.5, 1), figsize=(8, 4), logscale=False):
        '''
        Plot a snapshot of the internal flow.

        Parameters
        ----------
        snaps_array : numpy.ndarray
            Snapshots of the internal flow (also external, not only
            self.snapshots['internal']).
            The shape should be (nparams, npoints).
        idx : int, optional
            Index of the snapshot to plot. The default is 0.
        title : str, optional
            Title of the plot. The default is 'Snapshot'.
        namefig : str, optional
            Name of the file where the plot is saved. The default is None.
        lim_x : tuple, optional
            Limits of the x-axis. The default is (-0.5, 2).
        lim_y : tuple, optional
            Limits of the y-axis. The default is (-0.5, 1).
        figsize: tuple, optional
            Size of the output figure. The default is (8, 4).
        '''
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        snapshot = snaps_array[idx, :]
        if logscale:
            lognorm = matplotlib.colors.LogNorm(vmin=snapshot.min()+1e-12,
                vmax=snapshot.max())
            c = ax.tripcolor(self.auxiliary_triang, snapshot, cmap='rainbow',
                shading='gouraud', norm=lognorm)
        else:
            c = ax.tripcolor(self.auxiliary_triang, snapshot, cmap='rainbow',
                shading='gouraud')
        ax.plot(self._coords_airfoil()[0], self._coords_airfoil()[1],
                color='black', lw=0.5)
        ax.plot(self._coords_airfoil(which='neg')[0],
                self._coords_airfoil(which='neg')[1],
                color='black', lw=0.5)
        ax.set_aspect('equal')
        if lim_x is not None:
            ax.set_xlim(lim_x)
        if lim_y is not None:
            ax.set_ylim(lim_y)
        if title is not None:
            ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size= "5%", pad=0.1)
        plt.colorbar(c, cax=cax)
        if namefig is not None:
            fig.savefig(namefig, dpi=300)
        plt.show()

    def plot_airfoil(self, snaps_array, idx=0, title='Snapshot', namefig=None,
            lim_x=(-0.1, 1.1), lim_y=(-0.2, 0.2), figsize=(8, 4)):
        '''
        Plot a snapshot on the airfoil.

        Parameters
        ----------
        snaps_array : numpy.ndarray
            Snapshot on the airfoil (also external, not only
            self.snapshots['airfoil']).
            The shape should be (nparams, npoints).
        idx : int, optional
            Index of the snapshot to plot. The default is 0.
        title : str, optional
            Title of the plot. The default is 'Snapshot'.
        namefig : str, optional
            Name of the file where the plot is saved. The default is None.
        lim_x : tuple, optional
            Limits of the x-axis. The default is (-0.1, 1.1).
        lim_y : tuple, optional
            Limits of the y-axis. The default is (-0.2, 0.2).
        figsize: tuple, optional
            Size of the output figure. The default is (8, 4).
        '''
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        snapshot = snaps_array[idx, :]
        coords = self.pts_coordinates['airfoil']
        ax.grid()
        c = ax.scatter(coords[0, :], coords[1, :], c=snapshot, s=10)
        ax.set_aspect('equal')
        if lim_x is not None:
            ax.set_xlim(lim_x)
        if lim_y is not None:
            ax.set_ylim(lim_y)
        if title is not None:
            ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size= "5%", pad=0.1)
        plt.colorbar(c, cax=cax)
        if namefig is not None:
            fig.savefig(namefig, dpi=300)
        plt.show()

