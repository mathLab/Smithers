from ..abstract_dataset import AbstractDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


class GraetzDataset(AbstractDataset):

    parametric = True
    time_dependent = False
    description = "graetz problem TODO"
    data_directory = os.path.join(os.path.dirname(__file__), 'graetz')

    def __init__(self):

        self.params = np.load(os.path.join(self.data_directory, 'params.npy'))
        self.snapshots = np.load(os.path.join(self.data_directory, 'snapshots.npy'))
        self.pts_coordinates = np.load(os.path.join(self.data_directory, 'coords.npy'))
        self.faces = np.load(os.path.join(self.data_directory, 'triangles.npy'))
        self.triang = mtri.Triangulation(self.pts_coordinates[0], self.pts_coordinates[1], self.faces)

    def plot(self, idx=0):
        plt.tripcolor(self.triang, self.snapshots[idx], shading='gouraud')
        plt.colorbar()
        plt.show()
