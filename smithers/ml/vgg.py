'''
Module focused on the implementation of VGG.
'''
import torch
import torch.nn as nn
import torchvision

from smithers.ml.utils import decimate


class VGG(nn.Module):
    '''
    VGG base convolutions to produce lower-level feature maps.
    As a model to construct the VGG class we are considering the one
    already implemented in Pytorch
    (https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg16)
    and the one than can be found in this Pytorch tutorial for Object Detection:
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection.
    
    :param list cfg: If None, returns the configuration of VGG16. Otherwise
        a list of numbers and string 'M', representing all the layers of
        the net (its configuration), where the numbers represent the number
        of filters for that convolutional layers(i.e. the features
        extracted) and 'M' stands for the max pool layer
    :param bool batch_norm: If True, perform batch normalization
    :param string/sequential classifier: If is equal to the string
        'standard', build the classical VGG16 classifier layers for a VGG16
         to be trained on a dataset such as ImageNet (images 3 x 300 x 300).
         If is equal to 'cifar', build the classifier for VGG16 trained on a
         dataset like CIFAR is built. (N.B. here images are 3 x 32 x 32).
         See for more details:
         - Shuying Liu and Weihong Deng.
         'Very deep convolutional neural network based image classification
          using small training sample size.'
          In Pattern Recognition (ACPR),2015 3rd IAPR Asian Conference on,
          pages 730–734. IEEE, 2015.
          DOI: 10.1109/ACPR.2015.7486599
          - https://github.com/geifmany/cifar-vgg
          - https://github.com/chengyangfu/pytorch-vgg-cifar10
          If is equal to 'ssd', build the classifier layers for the SSD300
          architecture.
          Otherwise if is a sequential container, the classifier correspond
          exactly to this.
     :param int num_classes: number of classes in your dataset.
     :param bool init_weights: If True, the weights must be inizialized,
          otherwise weights pretrained on a particular dataset (Imagenet,
          CIFAR10, a custom one,..) are loaded from the file given in
          pretrain_weights.
     :param str pretrain_weights: path to the file containing the pretrained
          weights. Default value is None. For more details on the structure
          required for the checkpoint file, see 'save_checkpoint' in utils.py
    '''
    def __init__(self,
                 cfg=None,
                 classifier='standard',
                 batch_norm=False,
                 num_classes=1000,
                 init_weights=True,
                 pretrain_weights=None):
        super(VGG, self).__init__()

        self.num_classes = num_classes
        available_classifier = {
            'standard':
            nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, self.num_classes),
            ),
            'cifar':
            nn.Sequential(nn.Linear(512, self.num_classes), ),
            'ssd':
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1))
        }

        if cfg is None:
            configuration = [
                64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M'
            ]
            if classifier == 'ssd':
                configuration[-1] = 'M3'
            self.configuration = configuration
        else:
            self.configuration = cfg
        self.features = self.make_layers(batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier_str = classifier
        if isinstance(classifier, str):
            self.classifier = available_classifier.get(classifier)
        else:
            self.classifier = classifier
        self.pretrain_weights = pretrain_weights
        if init_weights:
            self._initialize_weights()
        else:
            self.load_pretrained_layers(cfg)

    def make_layers(self, batch_norm=False):
        '''
        Construct the structure of the net (only the features part)
        starting from the configuration given in input.

        :param bool batch_norm: If True, perform batch normalization
        :return: sequential object containing the structure of the
            features part of the net
        :rtype: nn.Sequential
        '''
        layers = []
        in_channels = 3
        for k in range(len(self.configuration)):
            if self.configuration[k] == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
                ]
            elif self.configuration[k] == 'M3':
                layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
            else:
                conv2d = nn.Conv2d(in_channels,
                                   self.configuration[k],
                                   kernel_size=3,
                                   padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(self.configuration[k]),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = self.configuration[k]
        return nn.Sequential(*layers)

    def forward(self, image):
        '''
        Forward propagation.

        :param torch.Tensor image: images, a tensor of dimensions
            (N, 3, width, height), where N is the number of images given in
            input.
        :param str classifier: a string corresponding to the classifier you
            are using
        :return: lower-level feature maps conv4_3 and conv7 (final output
            of the net, i.e. a vector with the predictions for every class)
        :rtype: torch.Tensor
        '''
        x = self.features[:20](image)
        conv4_3 = x.clone().detach()
        x = self.features[20:](x)
        if self.classifier_str == 'standard':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            # or equivalently
            # x = x.view(x.size(0), -1)
        elif self.classifier_str == 'cifar':
            x = torch.flatten(x, 1)
        x = self.classifier(x)
        return conv4_3, x

    def _initialize_weights(self):
        '''
        Random inizialization of the weights and bias coefficients of
        the network
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_pretrained_layers(self, cfg):
        '''
        Loading pre-trained Layers.
        If cfg is None and we are considering the standard version of VGG16,
        the state of VGG16 pretrained on Imagenet will be loaded. See
        https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We can also load the state of VGG16 pretrained on other datasets by
        giving in input a file containing the pretrained weights
        (pretrain_weights).
        If we are creating a custom version of this net with n_classes!=1000 the
        classifier will be changed.
        If cfg is None and we are condidering SSD300, we are converting fc6
        and fc7 into convolutional layers, and subsample by decimation. See
        the original paper where SSD300 is implemented for further
        details:  'SSD: Single Shot Multibox Detector' by
        Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
        Scott Reed, Cheng-Yang Fu, Alexander C. Berg
        https://arxiv.org/abs/1512.02325
        DOI:        10.1007/978-3-319-46448-0_2

        :param list cfg: If None, returns the configuration of VGG16. Otherwise
             a list of numbers and string 'M', representing all the layers of
             the net (its configuration), where the numbers represent the number
             of filters for that convolutional layers(i.e. the features
             extraxted) and 'M' stands for the max pool layer
        :return: state_dict, a dictionary containing the state of the net
        :rtype: dict
        '''
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        if self.pretrain_weights is None:
            pretrained_net = torchvision.models.vgg16(pretrained=True)
        else:
            pretrained_net = torch.load(self.pretrain_weights,
                                        torch.device('cpu'))
           # pretrained_net = pretrained_net['model']
        pretrained_state_dict = pretrained_net.state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        #print(param_names)

        if cfg is None and (self.classifier_str == 'standard'
                            or self.classifier_str == 'cifar'):
            if self.num_classes != 1000:
                pretrained_net.classifier = self.classifier
                pretrained_state_dict = pretrained_net.state_dict()
            state_dict = pretrained_state_dict

        elif cfg is None and self.classifier_str == 'ssd':
            # Transfer conv. parameters from pretrained model to current model
            for i, param in enumerate(
                    param_names[:-4]):  # excluding conv6 and conv7 parameters
                state_dict[param] = pretrained_state_dict[
                    pretrained_param_names[i]]

            # Convert fc6, fc7 to convolutional layers, and subsample
            # (by decimation) to sizes of conv6 and conv7
            # fc6
            conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(
                4096, 512, 7, 7)
            conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
            state_dict['classifier.0.weight'] = decimate(
                conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
            state_dict['classifier.0.bias'] = decimate(conv_fc6_bias,
                                                       m=[4])  # (1024)
            # fc7
            conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(
                4096, 4096, 1, 1)
            conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
            state_dict['classifier.1.weight'] = decimate(
                conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
            state_dict['classifier.1.bias'] = decimate(conv_fc7_bias,
                                                       m=[4])  # (1024)

        else:
            raise RuntimeError(
                'Invalid choice for configuration and classifier in order\
                 to use the pretrained model')

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")

        return state_dict
