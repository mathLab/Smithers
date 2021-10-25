from unittest import TestCase
import types
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from smithers.ml.utils import get_seq_model, PossibleCutIdx, spatial_gradients, projection, forward_dataset

class Testutils(TestCase):
    def test_get_seq_model_01(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11',
                               pretrained=True)
        model.classifier_str = 'standard'
        seq_model = get_seq_model(model)
        assert isinstance(seq_model, nn.Sequential)
        self.assertEqual(len(seq_model), 30)
        self.assertEqual(len(seq_model), len(model.features) +
                         len(model.classifier) + 2)

    def test_get_seq_model_02(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',
                               pretrained=True)
        model.classifier_str = 'standard'
        seq_model = get_seq_model(model)
        assert isinstance(seq_model, nn.Sequential)
        self.assertEqual(len(seq_model), 40)
        self.assertEqual(len(seq_model), len(model.features) +
                         len(model.classifier) + 2)

    def test_possiblecutidx_01(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11',
                               pretrained=True)
        model.classifier_str = 'standard'
        seq_model = get_seq_model(model)
        cut_idx = PossibleCutIdx(seq_model)
        assert isinstance(cut_idx, list)
        self.assertEqual(len(cut_idx), 11)

    def test_possiblecutidx_02(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',
                               pretrained=True)
        model.classifier_str = 'standard'
        seq_model = get_seq_model(model)
        cut_idx = PossibleCutIdx(seq_model)
        assert isinstance(cut_idx, list)
        self.assertEqual(len(cut_idx), 16)

    def test_constructor_spatial_gradients_01(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',
                               pretrained=True)
        model.classifier_str = 'standard'
        seq_model = get_seq_model(model)
        pre_model = seq_model[:12]
        post_model = seq_model[12:]
        input_ = [(torch.rand(3, 224, 224), torch.IntTensor([5]))]
        grad = spatial_gradients(input_, pre_model, post_model)
        assert isinstance(grad, types.GeneratorType)

    def test_constructor_spatial_gradients_02(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',
                               pretrained=True)
        model.classifier_str = 'standard'
        seq_model = get_seq_model(model)
        pre_model = seq_model[:11]
        post_model = seq_model[11:]
        input1 = (torch.rand(3, 224, 224), torch.IntTensor([5]))
        input2 = (torch.rand(3, 224, 224), torch.IntTensor([9]))
        input3 = (torch.rand(3, 224, 224), torch.IntTensor([3]))
        inputs = [input1, input2, input3]
        gradients = []
        for grad in spatial_gradients(inputs, pre_model, post_model):
            gradients.append(grad)
        assert isinstance(gradients[0], np.ndarray)

    def test_spatial_gradients_01(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',
                               pretrained=True)
        model.classifier_str = 'standard'
        seq_model = get_seq_model(model)
        pre_model = seq_model[:11]
        post_model = seq_model[11:]
        input1 = (torch.rand(3, 224, 224), torch.IntTensor([4]))
        input2 = (torch.rand(3, 224, 224), torch.IntTensor([8]))
        input3 = (torch.rand(3, 224, 224), torch.IntTensor([1]))
        inputs = [input1, input2, input3]
        gradients = []
        for grad in spatial_gradients(inputs, pre_model, post_model):
            gradients.append(grad)
            self.assertEqual(grad.size, 256 * 56 * 56)

    def test_spatial_gradients_02(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',
                               pretrained=True)
        model.classifier_str = 'standard'
        seq_model = get_seq_model(model)
        pre_model = seq_model[:8]
        post_model = seq_model[8:]
        input1 = (torch.rand(3, 224, 224), torch.IntTensor([1]))
        input2 = (torch.rand(3, 224, 224), torch.IntTensor([2]))
        input3 = (torch.rand(3, 224, 224), torch.IntTensor([3]))
        inputs = [input1, input2, input3]
        gradients = []
        generator = spatial_gradients(inputs, pre_model, post_model)
        for i in range(2):
            grad = next(generator)
            gradients.append(grad)
        self.assertEqual(len(gradients), 2)

    def test_spatial_gradients_03(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11',
                               pretrained=True)
        model.classifier_str = 'standard'
        seq_model = get_seq_model(model)
        pre_model = seq_model[:9]
        post_model = seq_model[9:]
        input1 = (torch.rand(3, 224, 224), torch.IntTensor([6]))
        input2 = (torch.rand(3, 224, 224), torch.IntTensor([7]))
        input3 = (torch.rand(3, 224, 224), torch.IntTensor([0]))
        inputs = [input1, input2, input3]
        gradients = []
        generator = spatial_gradients(inputs, pre_model, post_model)
        for i in range(2, 3):
            grad = next(generator)
            gradients.append(grad)
        self.assertEqual(len(gradients), 1)

    def test_projection(self):
        inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        dataset = TensorDataset(inps, tgts)
        data_loader = DataLoader(dataset, batch_size=2, pin_memory=True)
        proj_mat = torch.rand(2400, 50)
        matrix = torch.rand(10, 2400)
        mat_red = projection(proj_mat, data_loader, matrix)
        self.assertEqual(list(mat_red.size()), [10, 50])

    def test_forwarddataset_01(self):
        inps = torch.arange(10 * 3 * 224 * 224,
                            dtype=torch.float32).view(10, 3, 224, 224)
        tgts = torch.arange(10, dtype=torch.float32)
        dataset = TensorDataset(inps, tgts)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',
                               pretrained=True)
        model.classifier_str = 'standard'
        data_loader = DataLoader(dataset, batch_size=2, pin_memory=True)
        out_model = forward_dataset(model, data_loader)
        self.assertEqual(list(out_model.size()), [10, 1000])

    def test_forwarddataset_02(self):
        inps = torch.arange(50 * 3 * 224 * 224,
                            dtype=torch.float32).view(50, 3, 224, 224)
        tgts = torch.arange(50, dtype=torch.float32)
        dataset = TensorDataset(inps, tgts)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',
                               pretrained=True)
        model.classifier_str = 'standard'
        seq_model = get_seq_model(model)
        pre_model = seq_model[:11]
        data_loader = DataLoader(dataset, batch_size=2, pin_memory=True)
        out_model = forward_dataset(pre_model, data_loader)
        self.assertEqual(list(out_model.size()), [50, 256 * 56 * 56])
