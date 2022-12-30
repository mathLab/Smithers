'''
Utilities for the construction of the reduced version of a
Neural Network.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import decomposition
from scipy import linalg



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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
    layers.

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


# RANDOM SVD

def randomized_range_finder(A, size, n_iter=5):
    A = A.to('cpu')
    Q = np.random.normal(size=(A.shape[1], size))
    
    for i in range(n_iter):
        Q, _ = linalg.lu(A @ Q, permute_l=True)
        Q, _ = linalg.lu(A.T @ Q, permute_l=True)
        
    Q, _ = linalg.qr(A @ Q, mode='economic')
    return Q

def randomized_svd(M, n_components, n_oversamples=10, n_iter=2):
    n_random = n_components + n_oversamples
    
    Q = torch.tensor(randomized_range_finder(M, n_random, n_iter),dtype = torch.float)
    # project M to the (k + p) dimensional space using the basis vectors
    M = torch.tensor(M, dtype = torch.float).to('cpu')

    B = Q.transpose(0, 1) @ M
    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)
    Uhat = torch.tensor(Uhat).to('cpu')
    del B
    U = Q @ Uhat
    
    return U[:, :n_components], s[:n_components], V[:n_components, :]

#  ACTIVE SUBSPACES

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



def projection(proj_mat, data_loader, matrix, device = device):
    '''
    Funtion that performs the projection onto a space (e.g. the reduced
    space) of a matrix.

    :param torch.Tensor proj_mat: projection matrix n_feat x n_red.dim.
    :param iterable data_loader: iterable object for loading the dataset.
        It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :param torch.Tensor matrix: matrix to project n_images x n_feat.
        Possible way to construct it using the function forward_dataset.
    :param torch.device device: device used to allocate the variables for
        the function.
    :return: reduced matrix n_images x n_red.dim
    :rtype: torch.Tensor
    '''
    proj_mat = proj_mat.to(device)
    matrix_red = torch.zeros(0).to(device)
    num_batch = len(data_loader)
    batch_old = 0
    for idx_, batch in enumerate(data_loader):
        if idx_ >= num_batch:
            break

        images = batch[0].to(device)

        with torch.no_grad():
            proj_data = (matrix[batch_old : batch_old + images.size()[0], : ] @ proj_mat).to(device)
            batch_old = images.size()[0]
        matrix_red = torch.cat([matrix_red, proj_data.to(device)])

    return matrix_red


def forward_dataset(model, data_loader, device = device, flattening = True):
    '''
    Forward of a model using the whole dataset, i.e. the forward is
    performed by splitting the dataset in batches in order to reduce
    the computational effort needed.

    :param nn.Sequential/nn.Module model: model.
    :param iterable data_loader: iterable object for loading the dataset.
        It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :param torch.device device: device used to allocate the variables for the function.
    :param bool flattening: used to state whether flattening is desired or not (e.g. flattening = False for HOSVD).
    :return: output of the model computed on the whole dataset with
        dimensions n_images x n_feat (corresponds to n_class for the last
        layer)
    :rtype: torch.Tensor
    '''
    out_model = torch.zeros(0).to(device)
    num_batch = len(data_loader)
    for idx_, batch in enumerate(data_loader):
        if idx_ >= num_batch:
            break
        images = batch[0].to(device)

        with torch.no_grad():
            outputs = model(images).to(device)
            if flattening:
                outputs = torch.squeeze(outputs.flatten(1)).detach()
        out_model = torch.cat([out_model.to(device), outputs.to(device)]).to(device)
    return out_model.to(device)


# HOSVD - tensorial approach
def tensor_projection(proj_matrices, data_loader, model, device):
    '''
    Funtion that performs the projection onto a space (e.g. the reduced
    space) of a tensor (more than 2 dimensions).

    :param list proj_matrices: list containing the projection matrices for
        each dimension of the tensor. For each dimension i, the related
        projection matrix has size (input_dim[i], red_dim_list[i])
    :param iterable data_loader: iterable object for loading the dataset.
        It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :param nn.Module/torch.Tensor model: model under consideration for computing its
        outputs (that has to be reduced) or matrix to project n_images x C X H X W.
        Possible way to construct it using the function forward_dataset.
    :param torch.device device: device used to allocate the variables for
        the function.
    :return: reduced tensor n_images x red_dim[1] x red_dim[2] x red_dim[3]
    :rtype: torch.Tensor
    '''
    out_model = torch.zeros(0).to(device)
    batch_old = 0
    for idx_, batch in enumerate(data_loader):
        images = batch[0].to(device)
        with torch.no_grad():
            if torch.is_tensor(model):
                outputs = model[batch_old : batch_old + images.size()[0], : ]
                batch_old = images.size()[0]
            else:
                outputs = model(images).to(device)
            outputs = project_multiple_observations(proj_matrices, outputs)
        out_model = torch.cat([out_model, outputs.to(device)]).to(device)
        del outputs
        #torch.cuda.empty_cache()
    snapshots_red = torch.squeeze(out_model.flatten(1)).detach()
    return snapshots_red


def tensor_reverse(tensor):
    """
    Function that reverses the directions of a tensor

    :param torch.Tensor A: the input tensor with dimensions (d_1,d_2,...,d_n)
    :return: input tensor with reversed dimensions (d_n,...,d_2,d_1)
    :rtype: torch.Tensor
    """
    incr_list = [i for i in range(len(tensor.shape))]
    incr_list.reverse()
    return torch.permute(tensor, tuple(incr_list))

def project_single_observation(proj_matrices, observation_tensor):
    for i, _ in enumerate(observation_tensor.shape):
        observation_tensor = torch.tensordot(proj_matrices[i], observation_tensor, ([1],[i]))
    return tensor_reverse(observation_tensor)

def project_multiple_observations(proj_matrices, observations_tensor):
    for i in range(len(proj_matrices)):
        observations_tensor = torch.tensordot(proj_matrices[i], observations_tensor, ([1],[i+1]))
    return tensor_reverse(observations_tensor)
    

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
    for _, m in model.named_modules():
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


def compute_loss(model, device, test_loader, is_print=True, topk=[1], features=None):
    '''
    Function that computes the top-k accuracy of model for dataset=test_loader
    :param nn.Module model: reduced net
    :param torch.device device: object representing the device on
        which a torch.Tensor is or will be allocated.
    :param iterable test_loader: iterable object, it load the dataset.
        It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :param bool is_print:
    :param list top_k
    :return float test_accuracy
    '''
    model.eval()
    model.to(device)
    test_loss = 0

    res = []
    maxk = max(topk)
    batch_old = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch = data.size()[0]
            if features is None:
                output = model(data)
            else:
                output = model(data, features=features[batch_old : batch_old + batch, :])
            batch_old = batch
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').item()  # sum up batch loss
            # torch.tok Returns the k largest elements of the given
            # input tensor along a given dimension.
            _, pred = torch.topk(output, maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
    test_loss /= len(test_loader.sampler)
    print('Test Loss', test_loss / len(test_loader.sampler))
    correct = torch.FloatTensor(res).view(-1, len(topk)).sum(dim=0)
    test_accuracy = 100. * correct / len(test_loader.sampler)
    for idx, k in enumerate(topk):
        print(' Top {}:  Accuracy: {}/{} ({:.2f}%)'.format(
            k, correct[idx], len(test_loader.sampler), test_accuracy[idx]))
        print('Test Loss:', test_loss)
    if len(topk) == 1:
        return test_accuracy[0]
    else:
        return test_accuracy

def train_kd(student,
        teacher,
        device,
        train_loader,
        optimizer,
        train_max_batch,
        alpha=0.0,
        temperature=1.,
        lr_decrease=None,
        epoch=1,
        features=None):
    '''
    Function that retrains the model with knowledge distillation
    when alpha=0, it reduced to the original training
    :param nn.Module student: reduced net
    :param nn.Module teacher: full net
    :param torch.device device: object representing the device on
        which a torch.Tensor is or will be allocated.    
    :param iterable train_loader: iterable object, it load the dataset for
        training. It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :param optimizer
    :param train_max_batch
    :param float alpha: regularization parameter. Default value set to 0.0, 
        i.e. when the training is reduced to the original one
    :param float temperature: temperature factor introduced. When T tends to 
        infinity all the classes have the same probability, whereas when T
        tends to 0 the targets become one-hot labels. Default value set to 1.
    :param lr_decrease:
    :param int epoch: epoch number
    :return: accuracy
    '''
    student.train()
    teacher.eval()
    student.to(device)
    teacher.to(device)
    correct = 0.0
    batch_old = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch = data.size()[0]
        optimizer.zero_grad()
        if features is None:
            output = student(data)
        else:
            output = student(data, features[batch_old : batch_old + batch, :])
        output_teacher = teacher(data)
        batch_old = batch

        # The Kullback-Leibler divergence loss measure
        loss = nn.KLDivLoss()(F.log_softmax(output / temperature, dim=1),F.softmax(output_teacher / temperature, dim=1)
                             )*(alpha*temperature*temperature) + \
                 F.cross_entropy(output, target) * (1. - alpha)

        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    print('Train Loss kd:', loss.item() / len(train_loader.sampler))
    train_loss_val = loss.item() / len(train_loader.sampler)
    accuracy = correct / len(train_loader.sampler) * 100.0
    if lr_decrease is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decrease
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (epoch) / (epoch + 1)
    return accuracy, train_loss_val



def save_checkpoint(epoch, model, path, optimizer):
    '''
    Save model checkpoint.
    :param scalar epoch: epoch number
    :param list model: list of constructed classes that compose our network
    :param str path: path to the checkpoint location
    :param torch.Tensor optimizer: optimizer chosen
    '''
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, path)

