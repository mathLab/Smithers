from unittest import TestCase
import torch
import torch.nn as nn
from smithers.ml.rednet import RedNet
from smithers.ml.fnn import FNN

class Testrednet(TestCase):
    def test_constructor_0(self):
        pre_model = nn.Sequential(nn.Linear(500,300), nn.Linear(300,200))
        proj_mat = torch.rand(200, 50)
        inout_map = FNN(50, 5, 10)
        rednet = RedNet(5, pre_model, proj_mat, inout_map)
        assert isinstance(rednet.inout_map, nn.Module)
        assert isinstance(rednet.proj_model, nn.Linear)
        assert isinstance(rednet.premodel, nn.Sequential)

    def test_constructor_1(self):
        pre_model = nn.Sequential(nn.Conv2d(500,200,3))
        proj_mat = torch.rand(200, 50)
        inout_map = FNN(50, 5, 10)
        rednet = RedNet(5, pre_model, proj_mat, inout_map)
        assert isinstance(rednet.inout_map, nn.Module)
        assert isinstance(rednet.proj_model, nn.Linear)
        assert isinstance(rednet.premodel, nn.Sequential)

# TO DO: MISSING TEST CONSTRUCTOR CASE PCE (LIST MODEL AND COEFF)

    def test_forward_0(self):
        pre_model = nn.Sequential(nn.Linear(500,300), nn.Linear(300,200))
        proj_mat = torch.rand(200, 50)
        inout_map = FNN(50, 5, 10)
        rednet = RedNet(5, pre_model, proj_mat, inout_map)
        input_net = torch.rand(12, 500)
        output_net = rednet(input_net)
        self.assertEqual(list(output_net.size()), [12, 5])

    def test_forward_1(self):
        pre_model = nn.Sequential(nn.Conv2d(500,200,3))
        proj_mat = torch.rand(200, 50)
        inout_map = FNN(50, 5, 10)
        rednet = RedNet(5, pre_model, proj_mat, inout_map)
        input_net = torch.rand(12, 500, 3, 3)
        output_net = rednet(input_net)
        self.assertEqual(list(output_net.size()), [12, 5])
