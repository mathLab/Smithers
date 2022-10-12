'''
Module focused on the implementation of Prediction Convolutional Layers
'''
import torch
import torch.nn as nn


class PredictionConvolutions(nn.Module):
    '''
    Convolutions to predict class scores and bounding boxes using lower and
    higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t
    each of the 8732 prior (default) boxes. The Encode bounding boxes
    (that are in center-size form) w.r.t. the corresponding prior boxes
    (that are in center-size form).

    The class scores represent the scores of each object class in each of the
    8732 bounding boxes located. A high score for 'background' = no object.

    See the original paper where SSD300 is implemented for further
    details:  'SSD: Single Shot Multibox Detector' by
    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
    Scott Reed, Cheng-Yang Fu, Alexander C. Berg
    https://arxiv.org/abs/1512.02325
    DOI:10.1007/978-3-319-46448-0_2

    :param int n_classes: number of different types of objects
    :param list cfg_tot: If None, it returns the configuration used in the
        original SSD300 paper, mentioned before. Otherwise a list where
        every element is a number representing all the filters applied in
        that convolutional layer.
        NOTE: These layers are exactly the layers selected in low_feats and
        aux_conv_feats, thus the dimensions in this list have to be consistent
        with that of those convolutional layers.
    :param list n_boxes: If None, returns the number of prior-boxes for each
        feature map as described in the original paper for SSD300, i.e.
        {'conv4_3': 4, 'conv7': 6, 'conv8_2': 6, 'conv9_2': 6,'conv10_2': 4,
        'conv11_2': 4}, where 4 prior-boxes implies we use 4 different
        aspect ratios, etc. Otherwise you need to provide a list containing
        the number of prior boxes associated to every feature map
    '''
    def __init__(self, n_classes, cfg_tot=None, n_boxes=None):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        if cfg_tot is None:
            cfg_tot = [512, 1024, 512, 256, 256, 256]
        self.cfg_tot = cfg_tot

        if n_boxes is None:
            n_boxes = [4, 6, 6, 6, 4, 4]
        self.n_boxes = n_boxes

        # Localization prediction convolutions (predict offsets w.r.t
        # prior-boxes)
        self.features_loc = self.make_predlayers('loc')
        # Class prediction convolutions (predict classes in
        # localization boxes)
        self.features_cl = self.make_predlayers('cl')
        # Initialize convolutions' parameters
        self.init_conv2d()

    def make_predlayers(self, task):
        '''
        Construct the structure of the net starting from the configuration
        given in input.

        :param str task: a string that describes the task you are requiring,
            i.e. localization (for the definition of the correct bounding
            box) or classification (of the object in the picture)
        :return: sequential object containing the structure of the net
        :rtype: nn.Sequential
        '''
        layers = []
        for l in range(len(self.cfg_tot)):
            if task == 'loc':
                layers += [
                    nn.Conv2d(self.cfg_tot[l],
                              self.n_boxes[l] * 4,
                              kernel_size=3,
                              padding=1)
                ]
            elif task == 'cl':
                layers += [
                    nn.Conv2d(self.cfg_tot[l],
                              self.n_boxes[l] * self.n_classes,
                              kernel_size=3,
                              padding=1)
                ]
            else:
                raise RuntimeError(
                    'The task assigned is not recognized by the network.')
        return nn.Sequential(*layers)

    def init_conv2d(self):
        '''
        Initialize convolutional parameters
        '''
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)

    def forward(self, low_feats, auxconv_feats):
        '''
        Forward propagation.

        :param list of tensors low_feats: list representing the output of
            VGG.forward(), thus containing the low-level features map.
            For example in the case of SSD300, they are represented by
            conv4_3 and conv7:
        - conv4_3_feats: conv4_3 feature map, a tensor of dimensions
         (N, 512, 38, 38)
        - conv7_feats: conv7 feature map, a tensor of dimensions
         (N, 1024, 19, 19)
        :param list of tensors auxconv_feats: list representing the output of
            AuxiliaryConvolutions.forward(), thus containing the auxiliary
            convolution feature maps.
        For example, in the case of SSD300, they are:
        - conv8_2_feats: conv8_2 feature map, a tensor of dimensions
         (N, 512, 10, 10)
        - conv9_2_feats: conv9_2 feature map, a tensor of dimensions
         (N, 256, 5, 5)
        - conv10_2_feats: conv10_2 feature map, a tensor of dimensions
         (N, 256, 3, 3)
        - conv11_2_feats: conv11_2 feature map, a tensor of dimensions
         (N, 256, 1, 1)
        :return: total_numberpriors locations and class scores for each image.
            In the case of SSD it will returns 8732 locations and class scores
            (i.e. w.r.t each prior box) for each image
        :rtype: torch.Tensor, torch.Tensor
        '''
        #batch_size: the total number of images we are using
        #in our setting this is represented by the first number in the shape of
        # all the features map (this number is equal in all of them)
        batch_size = low_feats[0].size(0)
        conv_feats = low_feats + auxconv_feats

        locs = []
        classes_scores = []

        for k in range(len(conv_feats)):
            # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
            loc_conv = self.features_loc[k](conv_feats[k])
            loc_conv = loc_conv.permute(0, 2, 3, 1).contiguous()
            #  to match prior-box order (after .view())
            # (.contiguous() ensures it is stored in a contiguous chunk
            # of memory, needed for .view() below)
            loc_conv = loc_conv.view(batch_size, -1, 4)
            locs.append(loc_conv)

            # Predict classes in localization boxes
            cl_conv = self.features_cl[k](conv_feats[k])
            cl_conv = cl_conv.permute(0, 2, 3, 1).contiguous()
            #  to match prior-box order (after .view())
            # (.contiguous() ensures it is stored in a contiguous chunk
            # of memory, needed for .view() below)
            cl_conv = cl_conv.view(batch_size, -1, self.n_classes)
            classes_scores.append(cl_conv)

        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of
        # the prior-boxes)
        locs = torch.cat(locs, dim=1)
        classes_scores = torch.cat(classes_scores, dim=1)

        return locs, classes_scores
