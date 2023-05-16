from ..abstract_dataset import AbstractDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


class LidCavity(AbstractDataset):

    parametric = True
    time_dependent = False
    description = "lid cavity problem for increasing top wall velocities"
    data_directory = os.path.join(os.path.dirname(__file__), 'lid_cavity')

    def __init__(self) -> None:


        params = np.load(os.path.join(self.data_directory, 'params.npy'))
        snapshots_u = np.load(os.path.join(self.data_directory, 'snapshots_u.npy'))
        snapshots_p = np.load(os.path.join(self.data_directory, 'snapshots_p.npy'))
        coordinates = np.load(os.path.join(self.data_directory, 'coordinates.npy'))
        triang = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])

        self.params = params.reshape((-1, 1))
        self.snapshots = {'mag(v)': snapshots_u, 'p': snapshots_p}
        self.triang = triang
        self.coordinates = coordinates
        self.faces = None

    def plot(self, idx=0, out='mag(v)'):
        plt.tripcolor(self.triang, self.snapshots[out][idx], shading='gouraud')
        plt.colorbar()
        plt.title(out)
        plt.show()
