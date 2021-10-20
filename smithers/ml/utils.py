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
    returns a sequential model

    :param nn.Module model: CNN chosen, for example VGG16
    :return: sequential formula of the
        model that has be given in input
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
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            cutidx.append(int(i))  #  Find the Linear or Conv2d Layer Idx
    return cutidx


def give_inputs(dataset, pre_model):
    '''
    Generator for computing the inputs for the reduction layer, i.e. the
    output of the pre-model.

    :param Dataset/list of tuples dataset: dataset containing the 
        images/data.
    :param nn.Sequential pre_model: Sequential container representing the
        pre-model, i.e. the model cut on the cut-off layer.
    :return: matrix of inputs
    :rtype: numpy.ndarray
    '''
    for data in dataset:
        input0 = data[0].unsqueeze(0)  #add dimension as first axis
        target = torch.tensor([data[1]])
        input_ = pre_model(input0)
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
        target = torch.tensor([data[1]])
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
