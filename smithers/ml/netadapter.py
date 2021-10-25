'''
Module focused on the reduction of the ANN and implementaion of the
training and testing phases.
'''

import torch
import torch.nn as nn

from smithers.ml.rednet import RedNet
from smithers.ml.fnn import FNN, training_fnn
from smithers.ml.utils import PossibleCutIdx, spatial_gradients, forward_dataset, projection
#from ATHENA.athena.active import ActiveSubspaces
from smithers.ml.pcemodel import PCEModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class NetAdapter():
    '''
    Class that handles the reduction of a pretrained ANN and implementation 
    of the training and testing phases.
    '''
    def __init__(self, cutoff_idx, red_dim, red_method, inout_method):
        '''    
        :param int cutoff_idx: value that identifies the cut-off layer
        :param int red_dim: dimension of the reduced space onto which we
            project the high-dimensional vectors
        :param str red_method: string that identifies the reduced method to 
            use, e.g. 'AS', 'POD'
        :param str inout_method: string the represents the technique to use for
            the identification of the input-output map, e.g. 'PCE', 'ANN'
        '''

        self.cutoff_idx = cutoff_idx
        self.red_dim = red_dim
        self.red_method = red_method
        self.inout_method = inout_method

    def _reduce_AS(self, pre_model, post_model, train_dataset):
        '''
        Function that performs the reduction using Active Subspaces (AS)
        :param nn.Sequential pre_model: sequential model representing
            the pre-model.
        :param nn.Sequential post_model: sequential model representing
            the pre-model.
        :param Dataset train_dataset: dataset containing the training
            images.
        :returns: tensor proj_mat representing the projection matrix
            for AS (n_feat x red_dim)
        :rtype: torch.Tensor
        '''
        grad = spatial_gradients(train_dataset, pre_model, post_model)
        asub = ActiveSubspaces(dim=self.red_dim, method='exact')
        asub.fit(gradients=grad)
        proj_mat = torch.tensor(asub.evects, dtype=input_type)
        
        return proj_mat
    
    def _reduce_POD(self, matrix_features):
        '''
        Function that performs the reduction using the Proper Orthogonal
        Decomposition (POD).
        :param torch.Tensor matrix_features: (n_images x n_feat) matrix
            containing the output of the pre-model that needs to be reduced. 
        :returns: tensor proj_mat representing the projection matrix 
            for POD (n_feat x red_dim).
        :rtype: torch.Tensor
        '''
        u,sigma,v = torch.svd(torch.transpose(matrix_features,0,1))
        proj_mat = u[:, :self.red_dim]

        return proj_mat

    def _reduce(self, pre_model, post_model, train_dataset, train_loader):
        '''
        Function that performs the reduction of the high dimensional
        output of the pre-model
        :param nn.Sequential pre_model: sequential model representing
            the pre-model.
        :param nn.Sequential post_model: sequential model representing
            the pre-model.
        :param Dataset train_dataset: dataset containing the training
            images.
        :param iterable train_loader: iterable object for loading the dataset.
            It iterates over the given dataset, obtained combining a
            dataset(images and labels) and a sampler.
        :returns: tensors matrix_red and proj_mat containing the reduced output
            of the pre-model (n_images x red_dim) and the projection matrix
            (n_feat x red_dim) respectively.
        :rtype: torch.tensor
    	'''
        #matrix_features = matrixize(pre_model, train_dataset, train_labels)
        matrix_features = forward_dataset(pre_model, train_loader)

        if self.red_method == 'AS':
            #code for AS
            proj_mat = self._reduce_AS(pre_model, post_model, train_dataset)

        elif self.red_method == 'POD':
            #code for POD
            proj_mat = self._reduce_POD(matrix_features)    	  	
            
        else:
            raise ValueError
        
        matrix_red = projection(proj_mat, train_loader, matrix_features)

        return matrix_red, proj_mat

    def _inout_mapping_FNN(self, matrix_red, train_labels, n_class):
        '''
        Function responsible for the creation of the input-output map using
        a Feedfoprward Neural Network (FNN).

        :param torch.tensor matrix_red: matrix containing the reduced output
       	    of the pre-model.
        :param torch.tensor train_labels: tensor representing the labels associated
            to each image in the train dataset.
        :param int n _class: number of classes that composes the dataset
        :return: trained model of the FNN
        :rtype: nn.Module 
        '''
        n_neurons = 20
        targets = list(train_labels)
        fnn = FNN(self.red_dim, n_class, n_neurons)
        epochs = 500
        training_fnn(fnn, epochs , matrix_red, targets)

        return fnn

    def _inout_mapping_PCE(self, proj_mat, post_model, train_loader, train_labels):
        '''
        Function responsible for the creation of the input-output map using
        the Polynomial Chaos Expansion method (PCE).
        
        :param torch.tensor proj_mat: projection matrix.
        :param nn.Sequential pre_model: sequential model representing
            the pre-model.
        :param nn.Sequential post_model: sequential model representing
            the pre-model.
        :param iterable train_loader: iterable object, it load the dataset for
            training. It iterates over the given dataset, obtained combining a
            dataset (images and labels) and a sampler.
        :param torch.tensor train_labels: tensor representing the labels associated
            to each image in the train dataset.
        :return: trained model of PCE layer and PCE coeff
        :rtype: list
        '''

        out_postmodel = forward_dataset(post_model, train_loader)
        mean = torch.mean(Z_train, 0).to(device)
        var = torch.std(Z_train, 0).to(device)

        PCE_model = PCEModel(mean, var)
        coeff, training_score_LR, training_score_labels = PCE_model.Training(\
           Z_train, out_postmodel, train_labels[:Z_train.shape[0]])
        PCE_coeff = torch.tensor(coeff, dtype=torch.float32).to(device)

        return [PCE_model, PCE_coeff]
 
    def _inout_mapping(self, matrix_red, n_class, proj_mat, pre_model, post_model, train_labels, train_loader):
        '''
        Function responsible for the creation of the input-output map.
        :param tensor matrix_red: matrix containing the reduced output
            of the pre-model.
        :param tensor train_labels: tensor representing the labels associated
            to each image in the train dataset
        :param int n _class: number of classes that composes the dataset
        :param tensor proj_mat:	projection matrix.
        :param nn.Sequential pre_model: sequential model representing
            the pre-model.
        :param nn.Sequential post_model: sequential model representing
            the pre-model.
        :param iterable train_loader: iterable object, it load the dataset for
            training. It iterates over the given dataset, obtained combining a
            dataset (images and labels) and a sampler.
        :param tensor train_labels: tensor representing the labels associated
            to each image in the train dataset.
        :return: trained model of FNN or list with the trained model of PCE and
            the corresponding PCE coefficients
        :rtype: nn.Module/list
        '''
        if self.inout_method == 'FNN':
            #code for FNN
            inout_map = self._inout_mapping_FNN(matrix_red, train_labels, n_class)

        elif self.inout_method == 'PCE':
            #code for PCE
            inout_map = self._inout_mapping_PCE(proj_mat, pre_model, post_model, train_loader, train_labels)

        else:
            raise ValueError

        return inout_map                        

    def reduce_net(self, input_network, train_dataset, train_labels, train_loader, n_class):
        '''
        Function that performs the reduction of the network
        :param nn.Sequential input_network: sequential model representing
            the input network. If the sequential model is not provided, but
            instead you have a nn.Module obj, see the function get_seq_model
            in utils.py.
        :param Dataset train_dataset: dataset containing the training
            images
        :param torch.Tensor train_labels: tensor representing the labels associated
            to each image in the train dataset
        :param iterable train_loader: iterable object for loading the dataset.
            It iterates over the given dataset, obtained combining a
            dataset(images and labels) and a sampler.
        :param int n _class: number of classes that composes the dataset
        :return: reduced net
        :rtype: nn.Module
        '''
        input_type = train_dataset.__getitem__(0)[0].dtype
        possible_cut_idx = PossibleCutIdx(input_network)
        cut_idxlayer = possible_cut_idx[self.cutoff_idx]
        pre_model = input_network[:cut_idxlayer].to(dtype=input_type)
        post_model = input_network[cut_idxlayer:].to(dtype=input_type)
        matrix_red, proj_mat = self._reduce(pre_model, post_model,
                                            train_dataset, train_loader)
        inout_map = self._inout_mapping(matrix_red, n_class, proj_mat, pre_model, post_model, train_labels, train_loader)
        reduced_net = RedNet(n_class, pre_model, proj_mat, inout_map)

        return reduced_net
