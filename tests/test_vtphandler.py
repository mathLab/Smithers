from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os

from smithers.io import VTPHandler as Handler

test_file = "tests/test_datasets/cube.vtp"


def test_points():
    data = Handler.read(test_file)
    np.testing.assert_array_almost_equal(data["points"][0], [-0.5] * 3)


def test_number_points():
    data = Handler.read(test_file)
    np.testing.assert_equal(data["points"].shape, (24, 3))


def test_cells():
    data = Handler.read(test_file)
    np.testing.assert_equal(data["cells"][5], [20, 21, 23, 22])


def test_write():
    data = Handler.read(test_file)
    data["points"] += 1.0
    Handler.write("test.vtp", data)
