from ..abstract_dataset import AbstractDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


class ElasticBlockDataset(AbstractDataset):

    parametric = True
    time_dependent = False
    description = "elastic block TODO"
    data_directory = os.path.join(os.path.dirname(__file__), 'elastic_block')

    def __init__(self):

        self.params = np.load(os.path.join(self.data_directory, 'params.npy'))
        u1, u2 = np.split(np.load(os.path.join(self.data_directory, 'snapshots.npy')), 2, axis=1)
        self.snapshots = {
            'u1': u1,
            'u2': u2,
            'mag(u)': np.sqrt(u1**2+u2**2)
        }
        self.pts_coordinates = np.load(os.path.join(self.data_directory, 'coords.npy'))
        self.faces = np.load(os.path.join(self.data_directory, 'triangles.npy'))
        self.triang = mtri.Triangulation(self.pts_coordinates[0], self.pts_coordinates[1], self.faces)

    def plot(self, idx=0, out='mag(u)'):
        plt.tripcolor(self.triang, self.snapshots[out][idx])
        plt.colorbar()
        plt.title(out)
        plt.show()
