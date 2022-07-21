from unittest import TestCase
import torch
import torch.nn as nn
from smithers.ml.fnn import FNN, training_fnn

class Testfnn(TestCase):
    def test_constructor(self):
        fnn = FNN(50, 10, 20)
        assert isinstance(fnn.fc1, nn.Linear)
        self.assertEqual(fnn.n_hid, 20)

    def test_forward(self):
        fnn = FNN(50, 40, 10)
        input_net = torch.rand(12, 50)
        out = fnn(input_net)
        self.assertEqual(list(out.size()), [12, 40])

    def test_training(self):
        fnn = FNN(30, 4, 40)
        input_net = torch.rand(12, 30)
        real_out = torch.rand(12, 4)
        training_fnn(fnn, 10, input_net, real_out)
        self.assertEqual(fnn.n_hid, 40)
