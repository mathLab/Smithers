from unittest import TestCase
import torch
import torch.nn as nn
import numpy as np
from smithers.ml.layers.pcemodel import PCEModel

input_ = torch.rand(120, 50)
mean = torch.mean(input_, 0)
var = torch.var(input_, 0)

class TestPECModel(TestCase):
    def test_constructor_pce_01(self):
        pce = PCEModel(mean, var)
        assert isinstance(pce, nn.Module)
        self.assertEqual(pce.d, 50)
        self.assertEqual(pce.p, 2)
        self.assertEqual(pce.device, 'cpu')
        assert isinstance(pce.var, torch.Tensor)
        assert isinstance(pce.mean, torch.Tensor)
        assert isinstance(pce.oneDbasis, torch.Tensor)
        assert isinstance(pce.idxset, torch.Tensor)

    def test_constructor_pce_02(self):
        pce = PCEModel(mean, var, 40, 4)
        assert isinstance(pce, nn.Module)
        self.assertEqual(pce.d, 40)
        self.assertEqual(pce.p, 4)
        assert isinstance(pce.oneDbasis, torch.Tensor)
        assert isinstance(pce.idxset, torch.Tensor)

    def test_normalbasis(self):
        pce = PCEModel(mean, var, 40, 2)
        B = pce.NormalBasis()
        assert isinstance(B, torch.Tensor)
        self.assertEqual(list(B.size()), [3, 3])

    def test_forward(self):
        pce = PCEModel(mean, var, 50, 3)
        input1 = torch.rand(120, 50)
        Phi = pce.forward(input1)
        assert isinstance(Phi, torch.Tensor)
        self.assertEqual(Phi.size()[0], 120)

    def test_training(self):
        pce = PCEModel(mean, var, 50, 3)
        input1 = torch.rand(120, 50)
        out = torch.rand(120, 1)
        label = torch.rand(120, 1)
        coeff, approx, score = pce.Training(input1, out, label)
        assert isinstance(coeff, np.ndarray)
        assert isinstance(approx, float)
        assert isinstance(score, float)

    def test_inference(self):
        pce = PCEModel(mean, var, 50, 4)
        input1 = torch.rand(120, 50)
        coeff = torch.rand(1, 120)
        inf_mat = pce.Inference(input1, coeff)
        assert isinstance(inf_mat, torch.Tensor)

