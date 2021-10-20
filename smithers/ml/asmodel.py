'''
Module focused on the implementation of the Active Spubspace(AS)
layer as presented in the paper
Chunfeng Cui, Kaiqi Zhang, Talgat Daulbaev, Julia Gusak,
Ivan Oseledets, and Zheng Zhang. "Active Subspace of Neural
Networks: Structural Analysis and Universal Attacks".
accepted by SIAM Journal on Mathematics of Data Science (SIMODS)
'''

from torch.autograd.gradcheck import zero_gradients, gradcheck
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from functools import partial

minrank = 400


class ASModel(nn.Module):
    '''
    Class that  handles the creation of the AS layer (projecting the
    input on the active subspace) as defined in Step 1 of
    Algorithm 3.1. in the paper:
    - Chunfeng Cui, Kaiqi Zhang, Talgat Daulbaev, Julia Gusak,
    Ivan Oseledets, and Zheng Zhang. "Active Subspace of Neural 
    Networks: Structural Analysis and Universal Attacks".
    accepted by SIAM Journal on Mathematics of Data Science (SIMODS)
    '''
    def __init__(self, V, r, device):
        '''
        :param tensor V: matrix V, containing the left singular vectors 
            in the SVD decompositiond of matrix G (matrix containing the 
            gradients of the function chosen to compute the covariance
            matrix (as described in section 3.4)
            size of V: (n_features, n_sampling_batch), where n_sampling_batch
            is the max number of samplings we have per batch (r_max) and
            n_features the number of features of the layer we are examining,
            i.e. the dimension of the input we are taking into account.
        :param int r: dimension oif trunctated SVD
        :param torch.device device: object representing the device on
            which a torch.Tensor is or will be allocated.
        '''
        super(ASModel, self).__init__()
        self.device = device
        self.r_max = V.shape[1]
        self.r = self.r_max
        self.V_full = V.to(device)
        self.V = self.V_full[:, :r].to(device)

    def change_r(self, r_new):
        '''
        Function that takes into account the change of r and thus the 
        columns of V we are considering
        :param int r_new: scalar representing the new value for r
        '''
        if r_new > self.r:
            raise ValueError('New AS dim is greater than the maximum!')
        self.r = r_new
        self.V = self.V_full[:, :r_new]  #.to(self.device)

    def forward(self, x):
        '''
        Function performing the projection of a tensor on the AS
        (represented by V, where we are considering only r columns, 
        not the full V)
        :param tensor x: tensor to be projected on AS, size (1,n_features)
        :return: tensor x: tensor x projected on the AS, size (1, self.r)
        '''
        x = x.view(x.size(0), -1)  #.to(self.device)
        x = x @ self.V
        return x


def get_ASModel_FD(model, train_loader, cut_layer, max_batch, r_max, device):
    '''
    Function that implements the frequent direction algorithm for computing
    the active subspace (as described in Algorithm 3.2 in the paper mentioned
    above)
    :param nn.Sequential model: sequential model of the	net in exam
    :param iterable train_loader: iterable object, it load the dataset for
        training. It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :param list cut_layer: list of indexes where the net will be cut off
    :param int max_batch: maximum number of batches in the train
        dataset 
    :param int r_max: dimension of truncated SVD
    :param torch.device device: object representing the device on
        which a torch.Tensor is or will be allocated.
    :return: defaultdict ASlayers, Sigmas: dictionaries containing the
        projection of the inputs (i.e. the outputs of the cut off layer)
        on the AS and the singular values of Sigma
    '''
    as_emb = streamASEmbedding(model, train_loader, device, max_batch)
    as_emb.forward_backward(cut_layer)
    ASlayers = defaultdict(list)
    Sigmas = defaultdict(list)
    for key in as_emb.fds.keys():
        #print('Starting to Build Layer:',key)
        s, Vt = as_emb.fds[key].get()
        ASlayers[key] = ASModel(Vt.t(), r_max, device)
        Sigmas[key] = s
        print('Finished Building AS model for layer:', key)
    return ASlayers, Sigmas


class streamASEmbedding():
    '''
    Subroutine used in get_ASModel_FD() to compute the forward and
    backpropagation of the pre-model (model cut on the cut-off layer)
    '''
    def __init__(self, model, loader, device, batch_count=1):
        '''
        :param nn.Sequential model: sequential model of the net in exam
        :param iterable loader: iterable object, it load the dataset.
            It iterates over the given dataset, obtained combining a
            dataset(images and labels) and a sampler. 
        :param torch.device device: object representing the device on
        which a torch.Tensor is or will be allocated.
        :param int batch_count: a counter for the batches. Default
            value set to 1.
        '''
        self.model = model
        self.loader = loader
        self.device = device
        self.batch_count = batch_count
        self.logg_every = 2

        self.model.to(self.device)
        self.activations = defaultdict(list)
        self.fds = defaultdict(list)

        self.d_rate = 0.8

    def forward_backward(self, names):
        '''
        Function performing forward and backward propagations of the
        pre-model.
        :param list cut_layer: listr of indexes where the net will be cut off
        '''
        self.model.train()
        if type(names) != list:
            m = self.model[names - 1]
            m.register_backward_hook(partial(self.save_activation, names - 1))
        else:
            for i, layer in enumerate(names):
                m = self.model[layer - 1]
                # m = self.model.features[layer-1]
                m.register_backward_hook(
                    partial(self.save_activation, layer - 1))

        for batch_idx, (data, target) in enumerate(self.loader):

            data, target = data.to(self.device), target.to(self.device)
            # forward propagation
            #            print(data.size())
            #            print(self.model)
            output = self.model(data)
            #output = self.model(data, 'cifar')
            #output = output[1]
            loss = F.nll_loss(output, target)
            loss.backward()

            ### Update Frequent directions
            self.fd_step()

            if batch_idx >= self.batch_count:
                break

    def fd_step(self):
        '''
        Function that perform a step of the frequent direction algorithm
        '''
        for key in self.activations:
            if key not in self.fds.keys():
                bs, n_features = torch.cat(self.activations[key], 0).shape

                self.fds[key] = FrequentDirections(n = n_features,\
                       d = min(minrank,int(n_features*self.d_rate)))
            else:
                for row in torch.cat(self.activations[key], 0):
                    self.fds[key].append(row)

    def save_activation(self, name, mod, grad_inp, grad_out):
        bs = grad_out[0].shape[0]
        self.activations[name] = [grad_out[0].view(bs, -1).cpu()]


#         self.activations[name].append(grad_out[0].view(bs, -1).cpu())


# Subroutine used in streamASEmbedding.fd_step()
class FrequentDirections():
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.m = 2 * self.d
        self._sketch = torch.zeros((self.m, self.n), dtype=torch.float32)

        self.nextZeroRow = 0

    def append(self, vector):
        if torch.nonzero(vector).size(0) == 0:
            return

        if self.nextZeroRow >= self.m:
            self.__rotate__()

        self._sketch[self.nextZeroRow, :] = vector
        self.nextZeroRow += 1

    def __rotate__(self):
        Vt, s, _ = randomized_svd(self._sketch.t(), self.d)
        Vt = Vt.t()
        sShrunk = torch.sqrt(s[:self.d]**2 - s[self.d - 1]**2)
        self._sketch[:self.d:, :] = torch.diag(sShrunk) @ Vt[:self.d, :]
        self._sketch[self.d:, :] = 0
        self.nextZeroRow = self.d

    def get(self, rotate=True, take_root=False):
        if rotate:
            Vt, s, _ = randomized_svd(self._sketch.t(), self.d)
            Vt = Vt.t()
            if take_root:
                return torch.diag(torch.sqrt(s[:self.d])) @ Vt[:self.d, :]
            else:
                #                 return np.diag(s[:self.d]) @ Vt[:self.d, :]
                return s[:self.d], Vt[:self.d, :]

        return self._sketch[:self.d, :]


# The randomized SVD algorithm
# It is faster than SVD in general
def randomized_svd(A, k):
    '''
    Computes randomized rank-k truncated SVD of matrix A:
    A ≈ U @ torch.diag(Sigma) @ V.t()
    Args:
        A: input torch.Tensor matrix
        k: rank
    Returns:
        U, Sigma, V
    '''
    device = A.device
    with torch.no_grad():
        m, n = A.shape
        Omega = torch.randn(n, k, device=device)
        Y = A @ Omega
        Q, R = torch.qr(Y)
        B = Q.t() @ A
        Uhat, Sigma, V = torch.svd(B)
        U = Q @ Uhat
        return U, Sigma, V.t()



def compute_Z_AS_space(AS_model,
                       pre_model,
                       post_model,
                       data_loader,
                       num_batch=10,
                       device='cpu',
                       features=None):
    '''
    Function that perform the construction of the dataset D for obtaining the
    AS direction: it computes the output of the AS layer (X_train), i.e. the
    projection of the outputs of the pre-model into the AS, and the output of
    the post-model (y_train)

    :param defaultdict AS_model: dictionary containing the projection on the AS
        of the inputs (i.e. the outputs of the cut off layer)
    :param nn.Sequential pre_model: Sequential container representing the
        pre-model, i.e. the model cut on the cut-off layer
    :param nn.Sequential post_model: Sequential container representing
        the post-model, i.e. the model after the cut-off layer.
    :param iterable data_loader: iterable object for loading the dataset.
        It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :param int num_batch: number of batches
    :param torch.device device: object representing the device on
        which a torch.Tensor is or will be allocated.
    :return: output AS-layer (output pre-model projected into the AS) and output
        of the post-model (i.e. of the full model)
    :rtype: torch.Tensor, torch.Tensor
    '''
    X_train = torch.zeros(0)
    y_train = torch.zeros(0)
    batch_old = 0
    for idx_, (batch, target) in enumerate(data_loader):
        if idx_ >= num_batch:
            break

        batch = batch.to(device)

        with torch.no_grad():
            x = pre_model(batch)
            if features is None:
                as_data = AS_model(x).cpu()
            else:
                as_data = AS_model(features[batch_old : batch_old + batch.size()[0], :]).cpu()
                batch_old = batch.size()[0]
            y_data = post_model(x).cpu()
        X_train = torch.cat([X_train, as_data.cpu()])
        y_train = torch.cat([y_train, y_data.cpu()])

    return X_train, y_train

