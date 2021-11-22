'''
Utilities for the construction of the reduced version of a
Neural Network
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def get_seq_model(model):
    '''
    Takes a model with model.features and model.classifier and
    returns a sequential model. If attribute self.classifier_str
    is not present, add this attribute to the model before running
    this function.

    :param nn.Module model: CNN chosen, for example VGG16.
    :return: sequential formula of the model that has be given in input.
    :rtype: nn.Sequential
    '''
    if list(model.classifier.children()):
        if model.classifier_str == 'cifar':
            seq_model = nn.Sequential(*(list(model.features.children()) +
                                        [nn.Flatten(1, -1)] +
                                        list(model.classifier.children())))
        else:  #if model.classifier_str == 'standard':
            seq_model = nn.Sequential(*(list(model.features.children()) +
                                        [nn.AdaptiveAvgPool2d((7, 7))] +
                                        [nn.Flatten(1, -1)]+
                                        list(model.classifier.children())))
    else:
        if model.classifier_str == 'cifar':
            seq_model = nn.Sequential(*(list(model.features.children()) +
                                        [nn.Flatten(1, -1)] +
                                        [model.classifier]))
        elif model.classifier_str == 'standard':
            seq_model = nn.Sequential(*(list(model.features.children()) +
                                        [nn.AdaptiveAvgPool2d((7, 7))] +
                                        [nn.Flatten(1, -1)] +
                                        [model.classifier]))
    return seq_model



def PossibleCutIdx(seq_model):
    '''
    Function that identifies the possible indexes where the net can be
    cut, i.e. the indexes of the convolutional and fully-connected
    layers

    :param nn.Sequential seq_model: sequential container containing all
        the layers of the model
    :return: containing all the indexes of the convolutional and
        fully-connected layers
    :rtype: list
    '''
    cutidx = []
    for i, m in seq_model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):# or isinstance(m, nn.Conv2d):
            cutidx.append(int(i))  #  Find the Linear or Conv2d Layer Idx
    return cutidx


def give_inputs(dataset, model):
    '''
    Generator for computing the inputs for a layer, e.g the reduction
    layer.

    :param Dataset/list of tuples dataset: dataset containing the
        images/data.
    :param nn.Sequential model: Sequential container representing the
        model, e.g. the pre-model.
    :return: matrix of inputs/outputs of the model
    :rtype: numpy.ndarray
    '''
    for data in dataset:
        input0 = data[0].unsqueeze(0)  #add dimension as first axis
        #target = torch.tensor([data[1]])
        input_ = model(input0)
        yield torch.squeeze(input_.flatten(1)).detach().numpy()

def spatial_gradients(dataset, pre_model, post_model):
    '''
    Generator for computing the spatial gradients of the
    loss function composed with the postmodel (the derivative is
    computed w.r.t. the inputs of the AS, i.e. the evaluation of
    the premodel in the input of the net.

    :param Dataset/list of tuples dataset: dataset containing the
        images/data.
    :param nn.Sequential pre_model: Sequential container representing the
        pre-model, i.e. the model cut on the cut-off layer
    :param nn.Sequential post_model:  Sequential container representing
        the post-model, i.e. the model after the cut-off layer.
    :return: matrix of spatial gradients
    :rtype: numpy.ndarray
    '''
    for data in dataset:
        input0 = data[0].unsqueeze(0)  #add dimension as first axis
        target = torch.LongTensor([data[1]])
        input_ = pre_model(input0)
        out_post = post_model(input_)
        output_ = F.nll_loss(out_post, target, reduce=False)
        gradient = torch.autograd.grad(
            output_,
            input_,
            grad_outputs=torch.ones(output_.size()).to(dtype=input_.dtype,
                                                       device=input_.device),
            create_graph=True,
            retain_graph=True,
            allow_unused=True)
        yield torch.squeeze(gradient[0].flatten(1)).detach().numpy()


def matrixize(model, dataset, labels):
    '''
    Function that performs the construction of a matrix collecting
    the flatten output of the pre-model for each image of the
    dataset.
    :param nn.Sequential model: Sequential container representing the
        model we are using (e.g. the pre_model).
    :param Dataset dataset: dataset containing the images in exam.
    :param tensor labels: tensor representing the labels associated
        to each image in the dataset.
    :return: (n_features x n_images) matrix containing the output of
        the pre-model that needs to be reduced.
    :rtype: tensor
    '''
    matrix_features = torch.zeros(256 * 8 * 8, labels.size()[0])
    i = 0
    for data in dataset:
        input0 = data[0].unsqueeze(0)  #add dimension as first axis
        input_ = model(input0)
        matrix_features[:, i] = torch.squeeze(input_.flatten(1)).detach()
        i += 1
    return matrix_features


def projection(proj_mat, data_loader, matrix):
    '''
    Funtion that performs the projection onto a space (e.g. the reduced
    space) of a matrix.

    :param torch.Tensor proj_mat: projection matrix n_feat x n_red.dim.
    :param iterable data_loader: iterable object for loading the dataset.
        It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :param torch.Tensor matrix: matrix to project n_images x n_feat.
        Possible way to construct it using the function matrixise in
        utils.py.
    :return: reduced matrix n_images x n_red.dim
    :rtype: torch.Tensor
    '''

    matrix_red = torch.zeros(0)
    num_batch = len(data_loader)
    batch_old = 0
    for idx_, (batch, target) in enumerate(data_loader):
        if idx_ >= num_batch:
            break

        #batch = batch.to(device)

        with torch.no_grad():
            proj_data = (matrix[batch_old : batch_old +
                                batch.size()[0], :] @
                         proj_mat).cpu()
            batch_old = batch.size()[0]
        matrix_red = torch.cat([matrix_red, proj_data.cpu()])

    return matrix_red


def forward_dataset(model, data_loader):
    '''
    Forward of a model using the whole dataset, i.e. the forward is
    performed by splitting the dataset in batches in order to reduce
    the computational effort needed.

    :param nn.Sequential/nn.Module model: model.
    :param iterable data_loader: iterable object for loading the dataset.
        It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :return: output of the model computed on the whole dataset with
        dimensions n_images x n_feat (corresponds to n_class for the last
        layer)
    :rtype: torch.Tensor
    '''
    out_model = torch.zeros(0)
    num_batch = len(data_loader)
    for idx_, (batch, target) in enumerate(data_loader):
        if idx_ >= num_batch:
            break

        #batch = batch.to(device)

        with torch.no_grad():
            outputs = model(batch)
            outputs = torch.squeeze(outputs.flatten(1)).detach()
        out_model = torch.cat([out_model, outputs.cpu()])

    return out_model
    
def decimate(tensor, m):
    '''
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every
    'm'th value. This is used when we convert FC layers to equivalent
    Convolutional layers, but of a smaller size.
    :param torch.Tensor tensor: tensor to be decimated
    :param list m: list of decimation factors for each dimension of the
        tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    :rtype: torch.Tensor
    '''
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0,
                                                            end=tensor.size(d),
                                                            step=m[d]).long())

    return tensor
    
def Total_param(model, storage_per_param=4):
    '''
    Function that computes the total number of parameters

    :param nn.Module model: part of the net in exam
    :param int storage_per_param: memory needed to store a parameter.
        Default value set at 4.
    :return: total number of parameters
    :rtype: int
    '''
    total_params = 0
    for t in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += np.prod(t.data.cpu().numpy().shape)
    return total_params / 2**20 * storage_per_param
    
def Total_flops(model, device, is_ASNet=False, p=2, nAS=50):
    '''
    Function that computes the total number of flops

    :param nn.Module model: part of the net in exam
    :param torch.device device: object representing the device on
        which a torch.Tensor is or will be allocated.
    :param bool is_ASNet: Default value set at False.
    :param int p:
    :param int nAS: number of active neurons. Default value is
        set at 50.
    :return: total number of flops
    :rtype: float
    '''
    x = torch.ones([1, 3, 32, 32]).to(device)
    flops = 0.
    for i, m in model.named_modules():
        xold = x
        if isinstance(m, nn.MaxPool2d):
            x = m(x)
        if isinstance(m, nn.Conv2d):
            x = m(x)
            flops += xold.shape[1]*x.shape[1:].numel()*\
                    torch.tensor(m.kernel_size).prod()
        if isinstance(m, nn.Linear):
            flops += m.in_features * m.out_features

    if is_ASNet:
        flops += p * (model.PCE.in_features + nAS)  #Basis function
    return float(flops) / 10**6


