from ..abstract_dataset import AbstractDataset
import os
import numpy as np
import matplotlib.pyplot as plt


class UnsteadyHeatDataset(AbstractDataset):

    parametric = True
    time_dependent = True
    description = "Parametric heat transfer problem dataset"
    data_directory = os.path.join(os.path.dirname(__file__), 'unsteady_heat')

    def __init__(self):

        self.params = np.load(os.path.join(self.data_directory, 'params.npy'))
        self.snapshots = np.load(os.path.join(self.data_directory, 'snapshots.npy'))
        self.pts_coordinates = np.load(os.path.join(self.data_directory, 'coords.npy'))
        self.triang = np.load(os.path.join(self.data_directory, 'triangles.npy'))

    def plot(self, param_idx, time_instant, title='Unsteady heat'):
        x = self.pts_coordinates[:,0]
        y = self.pts_coordinates[:,1]
        plt.tripcolor(x, y, self.triang, self.snapshots[param_idx][time_instant])
        plt.colorbar()
        plt.title(title)
        plt.autoscale(tight=True)
        plt.show()
