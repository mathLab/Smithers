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
    def boundary(cls, data, axis=None):
        bd = np.concatenate(
            [
                np.min(data.vertices, axis=0)[None, :],
                np.max(data.vertices, axis=0)[None, :],
            ],
            axis=0,
        )
        if axis is None:
            return bd
        else:
            return bd[:, axis]

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
    def translate(cls, data, translation=[0, 0, 0]):
        """Move the object summing a 1D vector of X,Y,Z coordinates to its
        points.

        :param data: The OBJ data.
        :type data: WavefrontOBJ
        :param translation: A 1D vector which contains the value of the
            translation on X,Y,Z. Defaults to [0,0,0]
        :type translation: list, optional
        """
        data.vertices += translation

    @classmethod
    def rotate_around_axis(cls, data, axis, radians):
        """Rotate the object around the given axis. The rotation is performed
        in the direction given by the right-hand rule.

        The following rotates the object for 90 degrees around the Y axis:

        .. highlight:: python

            >>> ObjHandler.rotate_around_axis(data, [0,1,0], np.pi/2)

        :param data: The OBJ data.
        :type data: WavefrontOBJ
        :param axis: A 1D array which represents the vector around which the
            rotation is performed.
        :type scale: list
        :param radians: The amplitude of the rotation.
        :type radians: float
        """
        axis = np.array(axis) / np.linalg.norm(axis)
        axis *= radians

        data.vertices = Rotation.from_rotvec(axis).apply(data.vertices)

    @classmethod
    def switch_axes(cls, data, idx0, idx1):
        """Switch two coordinates.

        The following snippet switches X and Y axes:

        .. highlight:: python

            >>> ObjHandler.switch_axes(data, 0,1)

        :param data: The OBJ data.
        :type data: WavefrontOBJ
        :param idx0: The index of the first coordinate.
        :type scale: int
        :param idx1: The index of the second coordinate.
        :type scale: int
        """

        temp = np.array(data.vertices[:, idx0])
        data.vertices[:, idx0] = np.array(data.vertices[:, idx1])
        data.vertices[:, idx1] = temp

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
    def write(cls, data, filename):
        """Write the given instance of
        :class:`smithers.io.obj.objparser.WavefrontOBJ` to disk.

        :param data: The information to be put into the .obj file
        :type data: WavefrontOBJ
        :param filename: The output path
        :type filename: str
        """
        save_obj(data, filename)
