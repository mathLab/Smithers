'''
Module focused on the reduction of the ANN and implementation of the
training and testing phases.
'''

import torch
import torch.nn as nn
import numpy as np

from smithers.ml.models.rednet import RedNet
from smithers.ml.models.fnn import FNN, training_fnn
from smithers.ml.utils_rednet import PossibleCutIdx, spatial_gradients, forward_dataset, projection, tensor_projection, randomized_svd
from smithers.ml.layers.ahosvd import AHOSVD
from smithers.ml.layers.pcemodel import PCEModel

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
        :param int/list red_dim: dimension of the reduced space onto which we
            project the high-dimensional vectors or list of the reduced
            dimensions for each direction in the tensorial space under
            consideration.
        :param str red_method: string that identifies the reduced method to
            use, e.g. 'AS', 'POD'. 'AHOSVD'.
        :param str inout_method: string the represents the technique to use for
            the identification of the input-output map, e.g. 'PCE', 'FNN'.

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
        if isinstance(red_dim, list):
            self.red_dim_list = red_dim
            self.red_dim = np.prod(red_dim)
        else:
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

    def _reduce_HOSVD(self, model, data_loader, device): 
        '''
        Function that performs the reduction using the Higher
        order SVD (HOSVD) and in particular its averaged version (AHOSVD).

        :param nn.Module/torch.Tensor model: model under consideration for
           computing its outputs (that has to be reduced) or 
           (n_images x n_channel x H x W) tensor containing the output of
           the pre-model (in its tensorial version) that needs to be reduced.
        :param torch.device device: device used to allocate the variables for
            the function.
        :returns: list containing the projection matrices obtained via HOSVD
            for each dimension of the tensor (excluded the one related to the
            batch of images).
        :rtype: list
        '''
        batch_hosvd = 1
        batch_old = 0
        ahosvd = AHOSVD(torch.zeros(0), self.red_dim_list, batch_hosvd)
        for idx_, batch in enumerate(data_loader):
            images = batch[0].to(device)
        
            with torch.no_grad():
                if torch.is_tensor(model):
                    outputs = out[batch_old : batch_old + images.size()[0], : ]
                    batch_old = images.size()[0]
                else:
                    outputs = model(images).to(device)
                ahosvd_temp = AHOSVD(outputs, self.red_dim_list, batch_hosvd)
                ahosvd_temp.compute_u_matrices()
                ahosvd_temp.compute_proj_matrices()                

                ahosvd.proj_matrices = ahosvd.incremental_average(ahosvd.proj_matrices,
                                                                  ahosvd_temp.proj_matrices,
                                                                  idx_)
                del ahosvd_temp
                del outputs
                #torch.cuda.empty_cache()
        return ahosvd.proj_matrices

    
    def _reduce(self, pre_model, post_model, train_dataset, train_loader, device = device):
        '''
        Function that performs the reduction of the high dimensional
        output of the pre-model.
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
            matrix_red = projection(proj_mat, train_loader, matrix_features)

        elif self.red_method == 'POD':
            #code for POD
            matrix_features = forward_dataset(pre_model, train_loader).to(device)
            proj_mat = self._reduce_POD(matrix_features)
            matrix_red = projection(proj_mat, train_loader, matrix_features)

        elif self.red_method == 'RandSVD': 
            #code for RandSVD
            matrix_features = forward_dataset(pre_model, train_loader).to(device)
            proj_mat = self._reduce_RandSVD(matrix_features)
            matrix_red = projection(proj_mat, train_loader, matrix_features)

        elif self.red_method == 'HOSVD': 
            #code for HOSVD
            #tensor_features = forward_dataset(pre_model, train_loader, flattening = False).to(device)
            proj_mat = self._reduce_HOSVD(pre_model, train_loader, device)
            matrix_red = tensor_projection(proj_mat, train_loader, pre_model, device)

        else:
            raise ValueError

        return matrix_red, proj_mat

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

    def _inout_mapping(self, matrix_red, n_class, model, train_labels,
                       train_loader):
        '''
        Function responsible for the creation of the input-output map.
        :param tensor matrix_red: matrix containing the reduced output
            of the pre-model.
        :param int n _class: number of classes that composes the dataset
        :param tensor train_labels: tensor representing the labels associated
            to each image in the train dataset
        :param nn.Sequential model: sequential model representing
        :param tensor train_labels: tensor representing the labels associated
            to each image in the train dataset.
        :param iterable train_loader: iterable object, it load the dataset for
            training. It iterates over the given dataset, obtained combining a
            dataset (images and labels) and a sampler.
        :return: trained model of FNN or list with the trained model of PCE and
            the corresponding PCE coefficients
        :rtype: nn.Module/list
        '''
        if self.inout_method == 'FNN':
            #code for FNN
            inout_map = self._inout_mapping_FNN(matrix_red, train_labels, n_class) 

        elif self.inout_method == 'PCE':
            #code for PCE
            out_model = forward_dataset(model, train_loader)
            inout_map = self._inout_mapping_PCE(matrix_red, out_model, train_loader, train_labels)
        
        elif self.inout_method == None: 
            # In the case of object detection, we do not need this input_output part, since the
            # predictor is unchanged w.r.t. the original input network.
            inout_map = nn.Identity()

        else:
            raise ValueError

        return inout_map

    def reduce_net(self, input_network, train_dataset, train_labels,
                   train_loader, n_class, device = device):
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
        input_type = train_dataset.__getitem__(0)[0].dtype
        possible_cut_idx = PossibleCutIdx(input_network)
        cut_idxlayer = possible_cut_idx[self.cutoff_idx]
        pre_model = input_network[:cut_idxlayer].to(device, dtype=input_type)
        post_model = input_network[cut_idxlayer:].to(device, dtype=input_type)
        snapshots_red, proj_mat = self._reduce(pre_model, post_model, train_dataset, train_loader, device)
        inout_map = self._inout_mapping(snapshots_red, n_class, input_network, train_labels, train_loader)
        reduced_net = RedNet(n_class, pre_model, proj_mat, inout_map)
        return reduced_net.to(device)
