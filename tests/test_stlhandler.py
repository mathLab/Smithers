from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os

from smithers.io import STLHandler

stl_file = "tests/test_datasets/cube.stl"


def test_points():
    data = STLHandler.read(stl_file)
    np.testing.assert_array_almost_equal(data["points"][0], [-0.5] * 3)


def test_number_points():
    data = STLHandler.read(stl_file)
    np.testing.assert_equal(data["points"].shape, (8, 3))


def test_cells():
    data = STLHandler.read(stl_file)
    np.testing.assert_equal(data["cells"][5], [6, 1, 4])


def test_write():
    data = STLHandler.read(stl_file)
    data["points"] += 1.0
    STLHandler.write("test.stl", data)
