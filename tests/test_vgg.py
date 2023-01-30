from unittest import TestCase
import torch.nn as nn
import torchvision.transforms.functional as FT

from smithers.ml.models.vgg import VGG

default_cfg = [
    64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
    512, 512, 'M'
]  #vgg16
vgg19_cfg = [
    64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
    'M', 512, 512, 512, 512, 'M'
]

n_classes = 1000


class Testvgg(TestCase):
    def test_constuctor_empty(self):
        net = VGG()

    def test_make_layers_vgg16(self):
        net = VGG()
        layers = net.make_layers()
        assert len(layers) == 31

    def test_make_layers_conv(self):
        net = VGG()
        layers = net.make_layers()
        assert isinstance(layers[0], nn.Conv2d)

    def test_make_layers_vgg16bn(self):
        net = VGG()
        layers = net.make_layers(True)
        assert len(layers) == 44

    def test_make_layers_batch(self):
        net = VGG()
        layers = net.make_layers(True)
        assert isinstance(layers[1], nn.BatchNorm2d)

    def test_make_layers_maxp(self):
        net = VGG()
        layers = net.make_layers(True)
        assert isinstance(layers[6], nn.MaxPool2d)

    def test_constructor_arg(self):
        net = VGG(vgg19_cfg, 'ssd', True, init_weights='random')

    def test_vgg16_cifar(self):
        net = VGG(classifier='cifar')

    def test_ssd300(self):
        net = VGG(classifier='ssd', init_weights='imagenet')

    def test_init_weights(self):
        net = VGG(init_weights='random')
        params = list(net.parameters())
        assert len(params) == 32

    def test_init_weights_conv(self):
        net = VGG(init_weights='random')
        params = list(net.parameters())
        assert list(params[0].size()) == [64, 3, 3, 3]

    def test_init_weights_bias(self):
        net = VGG(init_weights='random')
        params = list(net.parameters())
        print(list(params[25].size()))
        assert list(params[25].size()) == [512]

    def test_load_pretrained_weights_01(self):
        net = VGG(init_weights='imagenet')
        state_dict = net.load_pretrained_layers(None)
        assert list(state_dict['features.0.bias'].size()) == [64]
        assert list(state_dict['classifier.6.weight'].size()) == [1000, 4096]

    def test_load_pretrained_weights_02(self):
        net = VGG(classifier='ssd', init_weights='imagenet')
        state_dict = net.load_pretrained_layers(None)
        assert list(
            state_dict['classifier.0.weight'].size()) == [1024, 512, 3, 3]

    def test_load_pretrained_weights_03(self):
        with self.assertRaises(RuntimeError):
            VGG(vgg19_cfg, 'ssd', init_weights='imagenet')

    def test_load_pretrained_weights_04(self):
        net = VGG(init_weights='imagenet', num_classes=4)
        state_dict = net.load_pretrained_layers(None)
        assert list(state_dict['features.0.bias'].size()) == [64]
        assert list(state_dict['classifier.6.weight'].size()) == [4, 4096]

