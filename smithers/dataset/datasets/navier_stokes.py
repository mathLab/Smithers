from ..abstract_dataset import AbstractDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


class NavierStokesDataset(AbstractDataset):

    parametric = True
    time_dependent = False
    description = "back step navier stokes TODO"
    data_directory = os.path.join(os.path.dirname(__file__), 'navier_stokes')

    def __init__(self):

        self.params = np.load(os.path.join(self.data_directory, 'params.npy'))
        vx, vy, p = np.split(
                np.load(os.path.join(self.data_directory, 'snapshots.npy')), 
                3,
                axis=1)

        self.snapshots = {
            'vx': vx,
            'vy': vy,
            'mag(v)': np.sqrt(vx**2+vy**2), 
            'p': p
        }
        self.pts_coordinates = np.load(os.path.join(self.data_directory, 'coords.npy'))
        self.faces = np.load(os.path.join(self.data_directory, 'triangles.npy'))
        self.triang = mtri.Triangulation(
                self.pts_coordinates[0], self.pts_coordinates[1], self.faces)

    def plot(self, idx=0, out='mag(v)'):
        plt.tripcolor(self.triang, self.snapshots[out][idx], shading='gouraud')
        plt.colorbar()
        plt.title(out)
        plt.show()
