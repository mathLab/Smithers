from unittest import TestCase
import unittest
import numpy as np
import os
from filecmp import cmp

from smithers.io.obj import ObjHandler
from smithers.io.obj.objparser import save_obj, WavefrontOBJ

data = ObjHandler.read("tests/test_datasets/file.obj")


class TestStlHandler(TestCase):
    def test_vertices(self):
        expected = [
            [0.109625, 0.06, -0.0488084],
            [0.0488084, 0.06, -0.109625],
            [0.0488084, 0.06, -0.2],
        ]

        np.testing.assert_almost_equal(data.vertices, expected, decimal=7)

    def test_polygons(self):
        expected = [[0, 1, 2], [2, 1, 0]]

        np.testing.assert_almost_equal(data.polygons, expected, decimal=7)

    def test_regions(self):
        assert data.regions == ["inner", "outer"]
        assert data.regions_change_indexes[0] == (0, 0)
        assert data.regions_change_indexes[1] == (1, 1)

    def test_write(self):
        x = WavefrontOBJ()
        x.regions = ["inner", "outer"]
        x.regions_change_indexes = [(0,0),(1,1)]
        x.vertices = [
                [0.109625, 0.06, -0.0488084],
                [0.0488084, 0.06, -0.109625],
                [0.0488084, 0.06, -0.2],
            ]
        x.polygons = [[0, 1, 2], [2, 1, 0]]

        save_obj(x, '/var/tmp/data.obj')
        assert cmp('/var/tmp/data.obj', "tests/test_datasets/file.obj")
