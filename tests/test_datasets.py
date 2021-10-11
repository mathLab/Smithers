from unittest import TestCase
import matplotlib.pyplot as plt
import numpy as np
from smithers.dataset import UnsteadyHeatDataset

"""
Tests for all the implemented datasets
"""


def test_init():
    pass

def test_unsteady_heat():
    UnsteadyHeatDataset().plot(10,20)
