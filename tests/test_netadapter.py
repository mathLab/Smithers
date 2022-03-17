from unittest import TestCase
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from smithers.ml.netadapter import NetAdapter
from smithers.ml.utils import get_seq_model

inps = torch.arange(100 * 3 * 224 * 224,
                    dtype=torch.float32).view(100, 3, 224, 224)
tgts = torch.arange(100, dtype=torch.float32)
train_dat = TensorDataset(inps, tgts)
train_load = DataLoader(train_dat, batch_size=2, pin_memory=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model.classifier_str = 'standard'
seq_model = get_seq_model(model)


class TestNetAdapter(TestCase):
    def test_constructor(self):
        netadapter = NetAdapter(3, 50, 'AS', 'PCE')
        self.assertEqual(netadapter.cutoff_idx, 3)
        self.assertEqual(netadapter.red_dim, 50)
        self.assertEqual(netadapter.red_method, 'AS')
        self.assertEqual(netadapter.inout_method, 'PCE')


    def test_reducenet_01(self):
        netadapter = NetAdapter(11, 50, 'AZ', 'PCE')
        with self.assertRaises(ValueError):
            netadapter.reduce_net(seq_model, train_dat, tgts, train_load, 40)

    def test_reducenet_02(self):
        netadapter = NetAdapter(5, 50, 'POD', 'ANN')
        with self.assertRaises(ValueError):
            netadapter.reduce_net(seq_model, train_dat, tgts, train_load, 40)

    def test_reducenet_03(self):
        netadapter = NetAdapter(5, 30, 'POD', 'FNN')
        red_net = netadapter.reduce_net(seq_model, train_dat, tgts, train_load, 1000)
        assert isinstance(red_net, nn.Module)
        assert isinstance(red_net.proj_model, nn.Linear)
        assert isinstance(red_net.inout_map, nn.Module)

    def test_reducenet_04(self):
        netadapter = NetAdapter(5, 40, 'POD', 'FNN')
        red_net = netadapter.reduce_net(seq_model, train_dat, tgts, train_load, 1000)
        input_ = torch.arange(1 * 3 * 224 * 224,
                              dtype=torch.float32).view(1, 3, 224, 224)
        out = red_net(input_)
        self.assertEqual(list(out.size()), [1, 1000])

    def test_reducenet_05(self):
        netadapter = NetAdapter(5, 20, 'POD', 'PCE')
        red_net = netadapter.reduce_net(seq_model, train_dat, tgts, train_load, 1000)
        assert isinstance(red_net, nn.Module)
        assert isinstance(red_net.proj_model, nn.Linear)
        assert isinstance(red_net.inout_map, nn.Sequential)

    def test_reducenet_06(self):
        netadapter = NetAdapter(6, 15, 'POD', 'PCE')
        red_net = netadapter.reduce_net(seq_model, train_dat, tgts, train_load, 1000)
        input_ = torch.arange(1 * 3 * 224 * 224,
                              dtype=torch.float32).view(1, 3, 224, 224)
        out = red_net(input_)
        self.assertEqual(list(out.size()), [1, 1000])
