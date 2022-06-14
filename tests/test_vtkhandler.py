import numpy as np

from smithers.io import VTKHandler

poly_file = "tests/test_datasets/cube.vtk"
ugrid_file = "tests/test_datasets/cube_grid.vtk"


def test_polydata():
    data = VTKHandler.read(poly_file)
    np.testing.assert_array_almost_equal(data["points"][0], [-0.5] * 3)
    np.testing.assert_equal(data["points"].shape, (24, 3))
    np.testing.assert_equal(data["cells"][5], [20, 21, 23, 22])


def test_grid():
    data = VTKHandler.read(ugrid_file, fmt='unstructured')
    np.testing.assert_array_almost_equal(data["points"][-1], [10] * 3)
    np.testing.assert_array_almost_equal(data["points"][0], [0] * 3)
    np.testing.assert_equal(data["cells"][5], [5, 6, 17, 16, 126, 127, 138, 137])
