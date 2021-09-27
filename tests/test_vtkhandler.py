from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os

from smithers.io import VTKHandler

vtk_file = "tests/test_datasets/cube.vtk"


def test_points():
    data = VTKHandler.read(vtk_file)
    np.testing.assert_array_almost_equal(data["points"][0], [-0.5] * 3)


def test_number_points():
    data = VTKHandler.read(vtk_file)
    np.testing.assert_equal(data["points"].shape, (24, 3))


def test_cellss():
    data = VTKHandler.read(vtk_file)
    np.testing.assert_equal(data["cells"][5], [20, 21, 23, 22])
