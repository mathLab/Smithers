'''
Class that handles the construction of the reduced network composed
by the premodel, the POD layer and the final ANN
'''
import torch
import torch.nn as nn
import copy

class RedNet(nn.Module):
    def __init__(self, premodel=None, POD_mat=None, ANN=None, checkpoint=None):
        super(RedNet, self).__init__()
        if checkpoint is not None:
            rednet = torch.load(checkpoint, torch.device('cpu'))
            self.premodel = rednet['model'].premodel
            self.POD_model = rednet['model'].POD_model
            self.ANN = rednet['model'].ANN
        else:
            self.premodel = premodel
            if isinstance(POD_mat, nn.Linear):
                self.POD_model = POD_mat
            else:
                self.POD_model = nn.Linear(POD_mat.size()[0], POD_mat.size()[1], bias=False)
                self.POD_model.weight.data = copy.deepcopy(POD_mat).t()
            self.ANN = ANN

    def forward(self, x):
        x = self.premodel(x)
        x = x.view(x.size(0), -1)
        x = self.POD_model(x)
        x = self.ANN(x)

        return x
