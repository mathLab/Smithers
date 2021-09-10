import numpy as np
from enum import Enum, auto
from Ofpp.mesh_parser import FoamMesh


def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def project(points, vectors):
    return np.dot(points, vectors.T)


class Parser(Enum):
    POINTS = auto()
    FACES = auto()
    BOUNDARY = auto()
    OWNER = auto()


def read_mesh_file(filename, parser: Parser):
    """
    Parse mesh file.

    :param filename: Name of the file.
    :param parser: A function used to parse the mesh.
    :return: Required data.
    """
    if parser == Parser.POINTS:
        parser = FoamMesh.parse_points_content
    elif parser == Parser.FACES:
        parser = FoamMesh.parse_faces_content
    elif parser == Parser.BOUNDARY:
        parser = FoamMesh.parse_boundary_content
    elif parser == Parser.OWNER:
        parser = FoamMesh.parse_owner_neighbour_content
    else:
        raise ValueError("Invalid parser.")

    return FoamMesh.parse_mesh_file(filename, parser)
