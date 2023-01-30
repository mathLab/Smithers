from unittest import TestCase
import torch
import torch.nn as nn
from smithers.ml.models.rednet import RedNet
from smithers.ml.models.fnn import FNN

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Testrednet(TestCase):
    def test_constructor_0(self):
        pre_model = nn.Sequential(nn.Linear(500, 300), nn.Linear(300, 200))
        proj_mat = torch.rand(200, 50)
        inout_map = FNN(50, 5, 10)
        rednet = RedNet(5, pre_model, proj_mat, inout_map)
        assert isinstance(rednet.inout_map, nn.Module)
        assert isinstance(rednet.proj_model, nn.Linear)
        assert isinstance(rednet.premodel, nn.Sequential)

    def test_constructor_1(self):
        pre_model = nn.Sequential(nn.Conv2d(500, 200, 3))
        proj_mat = torch.rand(200, 50)
        inout_map = FNN(50, 5, 10)
        rednet = RedNet(5, pre_model, proj_mat, inout_map)
        assert isinstance(rednet.inout_map, nn.Module)
        assert isinstance(rednet.proj_model, nn.Linear)
        assert isinstance(rednet.premodel, nn.Sequential)

# TO DO: MISSING TEST CONSTRUCTOR CASE PCE (LIST MODEL AND COEFF)
    def test_forward_0(self):
        pre_model = nn.Sequential(nn.Linear(650, 250), nn.Linear(250, 200))
        proj_mat = torch.rand(200, 50)
        inout_map = FNN(50, 5, 10)
        rednet = RedNet(5, pre_model, proj_mat, inout_map).to(device)
        input_net = torch.rand(20, 650).to(device)
        output_net = rednet(input_net)
        self.assertEqual(list(output_net.size()), [20, 5])

    def test_forward_1(self):
        pre_model = nn.Sequential(nn.Conv2d(600, 200, 3))
        proj_mat = torch.rand(200, 80)
        inout_map = FNN(80, 40, 50)
        rednet = RedNet(5, pre_model, proj_mat, inout_map).to(device)
        input_net = torch.rand(120, 600, 3, 3).to(device)
        output_net = rednet(input_net)
        self.assertEqual(list(output_net.size()), [120, 40])
