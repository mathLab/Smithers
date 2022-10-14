'''
Module focused on the reduction of the ANN and implementaion of the
training and testing phases.
'''

import torch
import numpy as np

from smithers.ml.rednet import RedNet
from smithers.ml.fnn import FNN, training_fnn
from smithers.ml.tensor_product_layer import tensor_product_layer
from smithers.ml.utils import PossibleCutIdx, spatial_gradients, forward_dataset, projection
from smithers.ml.utils import randomized_svd
from smithers.ml.AHOSVD import AHOSVD
from smithers.ml.pcemodel import PCEModel
#from ATHENA.athena.active import ActiveSubspaces

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

        :Example:

            >>> from smithers.ml.netadapter import NetAdapter
            >>> netadapter = NetAdapter(6, 50, 'POD', 'FNN')
            >>> original_network = import_net() # user defined method to load/build the original model
            >>> train_data = construct_dataset(path_to_dataset)
            >>> train_loader = load_dataset(train_data)
            >>> train_labels = train_data.targets
            >>> n_class = 10
            >>> red_model = netadapter.reduce_net(original_network, train_data, train_labels, train_loader, n_class)
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
        input_type = train_dataset.__getitem__(0)[0].dtype
        grad = spatial_gradients(train_dataset, pre_model, post_model)
        asub = ActiveSubspaces(dim=self.red_dim, method='exact').to(device)
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
        u = torch.svd(torch.transpose(matrix_features, 0, 1))[0]
        proj_mat = u[:, :self.red_dim]

        return proj_mat

    def _reduce_RandSVD(self, matrix_features): 
        '''
        Function that performs the reduction using the Randomized SVD (RandSVD).
        :param torch.Tensor matrix_features: (n_images x n_feat) matrix
            containing the output of the pre-model that needs to be reduced.
        :returns: tensor proj_mat representing the projection matrix
            obtained via RandSVD (n_feat x red_dim).
        :rtype: torch.Tensor
        '''
        matrix_features = matrix_features.to('cpu')
        u, _, _ = randomized_svd(torch.transpose(matrix_features, 0, 1), self.red_dim)
        return u

    def _reduce_HOSVD(self, tensor_features, mode_list_batch): 
        '''
        Function that performs the reduction using the Higher order SVD (RandSVD).
        :param torch.Tensor matrix_features: (n_images x n_feat) matrix
            containing the output of the pre-model that needs to be reduced.
        :returns: tensor proj_mat representing the projection matrix
            obtained via RandSVD (n_feat x red_dim).
        :rtype: torch.Tensor
        '''
        ahosvd = AHOSVD(tensor_features, mode_list_batch, mode_list_batch[0])
        ahosvd.compute_u_matrices()
        ahosvd.compute_proj_matrices()
        return ahosvd.proj_matrices, ahosvd
    

    def _reduce(self, pre_model, post_model, train_dataset, train_loader, device = device, mode_list_batch = None):
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
        :param torch.device device: object representing the device on
            which a torch.Tensor is or will be allocated.
        :returns: tensors matrix_red and proj_mat containing the reduced output
            of the pre-model (n_images x red_dim) and the projection matrix
            (n_feat x red_dim) respectively.
        :rtype: torch.tensor
    	'''

        if self.red_method == 'AS':
            #code for AS
            matrix_features = forward_dataset(pre_model, train_loader).to(device)
            proj_mat = self._reduce_AS(pre_model, post_model, train_dataset)
            snapshots_red = projection(proj_mat, train_loader, matrix_features)

        elif self.red_method == 'POD':
            #code for POD
            matrix_features = forward_dataset(pre_model, train_loader).to(device)
            proj_mat = self._reduce_POD(matrix_features)
            snapshots_red = projection(proj_mat, train_loader, matrix_features)

        elif self.red_method == 'RandSVD': 
            #code for RandSVD
            matrix_features = forward_dataset(pre_model, train_loader).to(device)
            proj_mat = self._reduce_RandSVD(matrix_features)
            snapshots_red = projection(proj_mat, train_loader, matrix_features)

        elif self.red_method == 'HOSVD': 
            #code for HOSVD
            tensor_features = forward_dataset(pre_model, train_loader, flattening = False).to(device)
            proj_mat, ahosvd = self._reduce_HOSVD(tensor_features, mode_list_batch)
            snapshots_red = ahosvd.project_multiple_observations(tensor_features)
            # uncomment to get the shape of the reduced snapshots tensor
            # print(f"La shape di tensor_red è {snapshots_red.shape}", flush = True)
            snapshots_red = torch.squeeze(snapshots_red.flatten(1)).detach()
            

        else:
            raise ValueError

        return snapshots_red, proj_mat

    def _inout_mapping_FNN(self, matrix_red, train_labels, n_class):
        '''
        Function responsible for the creation of the input-output map using
        a Feedfoprward Neural Network (FNN).

        :param torch.tensor matrix_red: matrix containing the reduced output
       	    of the pre-model.
        :param torch.tensor train_labels: tensor representing the labels
            associated to each image in the train dataset.
        :param int n _class: number of classes that composes the dataset
        :return: trained model of the FNN
        :rtype: nn.Module
        '''
        n_neurons = 20
        targets = list(train_labels)
        fnn = FNN(self.red_dim, n_class, n_neurons).to(device)
        epochs = 500
        training_fnn(fnn, epochs, matrix_red.to(device), targets)

        return fnn

    def _inout_mapping_PCE(self, matrix_red, out_postmodel, train_loader,
                           train_labels):
        '''
        Function responsible for the creation of the input-output map using
        the Polynomial Chaos Expansion method (PCE).

        :param torch.tensor matrix_red: matrix containing the reduced output
            of the pre-model.
        :param nn.Sequential post_model: sequential model representing
            the pre-model.
        :param iterable train_loader: iterable object, it load the dataset for
            training. It iterates over the given dataset, obtained combining a
            dataset (images and labels) and a sampler.
        :param torch.tensor train_labels: tensor representing the labels
            associated to each image in the train dataset.
        :return: trained model of PCE layer and PCE coeff
        :rtype: list
        '''
        mean = torch.mean(matrix_red, 0).to(device)
        var = torch.std(matrix_red, 0).to(device)

        PCE_model = PCEModel(mean, var)
        coeff = PCE_model.Training(matrix_red, out_postmodel,
                                   train_labels[:matrix_red.shape[0]])[0]
        PCE_coeff = torch.FloatTensor(coeff).to(device)

        return [PCE_model, PCE_coeff]

    def _inout_mapping(self, matrix_red, n_class, out_model, train_labels,
                       train_loader):
        '''
        Function responsible for the creation of the input-output map.
        :param tensor matrix_red: matrix containing the reduced output
            of the pre-model.
        :param tensor train_labels: tensor representing the labels associated
            to each image in the train dataset
        :param int n _class: number of classes that composes the dataset
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
            inout_map = self._inout_mapping_PCE(matrix_red, out_model, train_loader, train_labels)

        else:
            raise ValueError

        return inout_map

    def reduce_net(self, input_network, train_dataset, train_labels,
                   train_loader, n_class, device = device, mode_list_batch = [25, 35, 3, 3]):
        '''
        Function that performs the reduction of the network
        :param nn.Sequential input_network: sequential model representing
            the input network. If the sequential model is not provided, but
            instead you have a nn.Module obj, see the function get_seq_model
            in utils.py.
        :param Dataset train_dataset: dataset containing the training
            images
        :param torch.Tensor train_labels: tensor representing the labels
            associated to each image in the train dataset
        :param iterable train_loader: iterable object for loading the dataset.
            It iterates over the given dataset, obtained combining a
            dataset(images and labels) and a sampler.
        :param int n _class: number of classes that composes the dataset
        :param torch.device device: object representing the device on
            which a torch.Tensor is or will be allocated.  
        :return: reduced net
        :rtype: nn.Module
        '''
        print('Initializing reduction. Chosen reduction method is: '+self.red_method, flush=True)
        input_type = train_dataset.__getitem__(0)[0].dtype
        possible_cut_idx = PossibleCutIdx(input_network)
        cut_idxlayer = possible_cut_idx[self.cutoff_idx]
        pre_model = input_network[:cut_idxlayer].to(device, dtype=input_type)
        post_model = input_network[cut_idxlayer:].to(device, dtype=input_type)
        if self.inout_method == 'PCE':
            out_model = forward_dataset(input_network, train_loader)
        snapshots_red, proj_mat = self._reduce(pre_model, post_model, train_dataset, train_loader, device, mode_list_batch)
        if self.inout_method == 'PCE':
            inout_map = self._inout_mapping(snapshots_red, n_class, out_model, train_labels, train_loader)
        else:
            inout_map = self._inout_mapping(snapshots_red, n_class, None, train_labels, train_loader)
        if self.red_method == 'HOSVD':
            proj_matrices_layer = tensor_product_layer(proj_mat)
            reduced_net = RedNet(n_class, pre_model, proj_matrices_layer, inout_map)
        else:
            reduced_net = RedNet(n_class, pre_model, proj_mat, inout_map)
        return reduced_net.to(device)