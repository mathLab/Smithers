'''
Module focused on the implementation of Auxiliary Convolutional Layers
'''
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    see the original paper where SSD300 is implemented for further
    details:  'SSD: Single Shot Multibox Detector' by
    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
    Scott Reed, Cheng-Yang Fu, Alexander C. Berg
    https://arxiv.org/abs/1512.02325
    DOI:10.1007/978-3-319-46448-0_2
    """
    def __init__(self, layers=None):
        """
        :param list layers: If None, returns the configuration used in the
            original SSD300 paper, mentioned before. Otherwise a list where
            every element is a list of numbers representing the number of
            filters for that convolutional layer
        """
        super(AuxiliaryConvolutions, self).__init__()

        if layers is None:
            layers = [[256, 512], [128, 256], [128, 256], [128, 256]]
        self.layers = layers
        self.features = self.make_auxlayers()
        #Inizialize convolutions' parameters
        self.init_conv2d()

    def make_auxlayers(self):
        """
        # Auxiliary/additional convolutions on top of the VGG base
        :param list cfg: configuration of the auxiliary layer for our CNN
            (number of filters applied in that layers (thus the features
            extracted))
        """
        layers = []
        in_channels = 1024  
        #1280 number to be changed, put as param function
        for k in range(len(self.layers)):
            layers += [
                nn.Conv2d(in_channels,
                          self.layers[k][0],
                          kernel_size=1,
                          padding=0)
            ]  #stride=1 by default
            if k < 2:
                layers += [
                    nn.Conv2d(self.layers[k][0],
                              self.layers[k][1],
                              kernel_size=3,
                              stride=2,
                              padding=1)
                ]
                # dim. reduction because stride > 1
            else:
                layers += [
                    nn.Conv2d(self.layers[k][0],
                              self.layers[k][1],
                              kernel_size=1,#3, #1 change
                              padding=0)
                ]
                # dim. reduction because padding=0
            in_channels = self.layers[k][1]

        return nn.Sequential(*layers)

    def init_conv2d(self):
        """
        Initialize convolution parameters
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.
        :param Tensor conv7_feats: output of last classification layer
            base network, lower-level conv7 feature map, a tensor of
            dimensions (N, 1024, 19, 19)in the case of VGG16
	Note: Since these layers are thought as additional layers to be placed
        after a base network, pay attention that the dimensions of conv7_feats
        have to be consistent with that of the first layer of this structure.
        :return list out_conv2: list containing higher-level feature maps
            conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out_in = conv7_feats
        out_conv2 = []
        for conv in self.features:
            out = F.relu(conv(out_in))
            out_conv2.append(out)
            out_in = out

        # Higher-level feature maps, only the elements on odd position
        # thus the features conv_2
        return out_conv2[1::2]
