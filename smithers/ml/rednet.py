'''
Class that handles the construction of the reduced network composed
by the premodel, the reduction layer and the final input-output map.
'''
import copy
import torch
import torch.nn as nn
from smithers.ml.tensor_product_layer import tensor_product_layer

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class RedNet(nn.Module):
    '''
    Creation of the reduced Neural Network starting from the
    different blocks that composes it: pre-model, projection
    matrix proj_mat and input-output mapping inout_map.

    :param int n_classes: number of classes that composes the dataset.
    :param nn.Sequential premodel: sequential model representing the pre-model.
        Default value set to None.
    :param torch.tensor proj_mat: projection matrix. Default value set to None.
    :param nn.Module/list inout_map: input-output mapping. For example it can be
        a trained model of FNN or a list with the trained model of PCE and the
        corresponding PCE coefficients. Default value set to None.
    :param path_file checkpoint: If None, you will use the previuos block to
        initialize the reduced model, otherwise you will load them from the
        checkpoint file given in input.
    '''
    def __init__(self, n_classes, premodel=None, proj_mat=None, inout_map=None,
                 checkpoint=None):
        super(RedNet, self).__init__()
        if checkpoint is not None:
            rednet = torch.load(checkpoint, torch.device('cpu'))
            self.premodel = rednet['model'].premodel
            self.proj_model = rednet['model'].proj_model
            self.inout_map = rednet['model'].inout_map
        else:
            self.premodel = premodel
            if isinstance(proj_mat, nn.Linear):
                self.proj_model = proj_mat
            elif isinstance(proj_mat, tensor_product_layer):
                self.proj_model = proj_mat
            else:
                self.proj_model = nn.Linear(proj_mat.size()[0],
                                            proj_mat.size()[1], bias=False)
                self.proj_model.weight.data = copy.deepcopy(proj_mat).t()

            if isinstance(inout_map, list):
                self.inout_basis = inout_map[0]
                self.inout_lay = nn.Linear(inout_map[0].nbasis, n_classes,
                                           bias=False)
                self.inout_lay.weight.data = copy.deepcopy(inout_map[1]).t()
                self.inout_map = nn.Sequential(self.inout_basis, self.inout_lay)
            else:
                self.inout_map = inout_map

    def forward(self, x):
        '''
        Forward Phase. The first clause concerns AHOSVD, the other one is a more general version.

        :param torch.tensor x: input for the reduced net with dimensions
            n_images x n_input.
        :return: output n_images x n_class
        :rtype: torch.tensor
        '''
        if isinstance(self.proj_model, tensor_product_layer):
            x = x.to(device)
            x = self.premodel(x)
            x = self.proj_model(x)
            if len(self.proj_model.list_of_matrices) == len(x.shape):
                x = x.flatten()
            elif len(x.shape) == len(self.proj_model.list_of_matrices) + 1:
                x = x.reshape(x.shape[0], int(torch.prod(torch.tensor(x.shape[1:]))))
            x = self.inout_map(x)
        else:    
            x = x.to(device)
            x = self.premodel(x)
            x = x.view(x.size(0), -1)
            x = self.proj_model(x)
            x = self.inout_map(x)

        return x
