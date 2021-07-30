from .objparser import load_obj, save_obj, WavefrontOBJ
import numpy as np
from scipy.spatial.transform import Rotation


class ObjHandler:
    """
    Handler for .obj files.
    """

    @classmethod
    def read(cls, filename):
        """Load an .obj file.

        :param filename: The path of the file.
        :type filename: str
        :returns: An object which holds the information contained in the
            file.
        :rtype: WavefrontOBJ
        """

        return load_obj(filename)

    @classmethod
    def scale(cls, data, scale=[1, 1, 1]):
        """Scale the position of the vertices in the given `data` variable
        using the given scaling vector.

        :param data: The OBJ data.
        :type data: WavefrontOBJ
        :param scale: A 1D vector which contains the scaling factors for each
            component, defaults to [1,1,1]
        :type scale: list
        """
        data.vertices = data.vertices * scale

    @classmethod
    def rotate_around_axis(cls, data, axis, radians):
        """Scale the position of the vertices in the given `data` variable
        using the given scaling vector.

        :param data: The OBJ data.
        :type data: WavefrontOBJ
        :param scale: A 1D vector which contains the scaling factors for each
            component, defaults to [1,1,1]
        :type scale: list
        """
        axis = np.array(axis) / np.linalg.norm(axis)
        axis *= radians

        data.vertices = Rotation.from_rotvec(axis).apply(
            data.vertices
        )

    @classmethod
    def dimension(cls, data):
        """Evaluate the dimension (in each direction) of the object represented
        by the given .obj file (encapsulated into an object of type
        :class:`smithers.io.obj.objparser.WavefrontOBJ`).

        :param data: The .obj file.
        :type data: WavefrontOBJ
        :return: The dimension of the object represented by the given file.
        :rtype: np.ndarray
        """

        return np.max(data.vertices, axis=0) - np.min(data.vertices, axis=0)

    @classmethod
    def write(cls, filename, data):
        """Write the given instance of
        :class:`smithers.io.obj.objparser.WavefrontOBJ` to disk.

        :param filename: The output path
        :type filename: str
        :param data: The information to be put into the .obj file
        :type data: WavefrontOBJ
        """
        save_obj(data, filename)
