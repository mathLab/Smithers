'''
Class that handles the creation of a Feedforward Neural
Network (FNN).
'''

from numpy import real
import torch
import torch.nn as nn
import torch.optim as optim
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class FNN(nn.Module):
    def __init__(self, n_input, n_output, inner_size=20,
                 n_layers=1, func=nn.Softplus, layers=None):
        '''
        Construction of a Feedforward Neural Network (FNN) with
        given a number of input and output neurons, one hidden
        layer with a number n_hid of neurons.

        :param int n_input: number of input neurons
        :param int n_output: number of output neurons that corresponds
            to the number of classes that compose the dataset
        :param int inner_size: number of hidden neurons
        :param int n_layers: number of hidden layers
        :param nn.Module func: activation function. Default
            function: nn.Softplus
        :param list layers: list where each component represents the
            number of hidden layers for the corresponding layer
        :param bool cifar: boolean to identify if we are using as dataset
            the cifar one
        '''
        super(FNN, self).__init__()

        self.n_input = n_input
        self.n_output = n_output

        if layers is None:
            layers = [inner_size] * n_layers

        tmp_layers = layers.copy()
        tmp_layers.insert(0, self.n_input)
        tmp_layers.append(self.n_output)
        #tmp_layers[0] = self.n_input


        self.layers = []
        for i in range(len(tmp_layers)-1):
            self.layers.append(nn.Linear(tmp_layers[i], tmp_layers[i+1]))

        if isinstance(func, list):
            self.functions = func
        else:
            self.functions = [func for _ in range(len(self.layers)-1)]

        if len(self.layers) != len(self.functions) + 1:
            raise RuntimeError('uncosistent number of layers and functions')


        unique_list = []
        for layer, func in zip(self.layers[:-1], self.functions):
            unique_list.append(layer)
            if func is not None:
                unique_list.append(func())
        unique_list.append(self.layers[-1])

        self.model = nn.Sequential(*unique_list)


    def forward(self, x):
        '''
        Forward Phase.

        :param tensor x: input of the network with dimensions
            n_images x n_input
        :return: output of the FNN n_images x n_output
        :rtype: tensor
        '''
        return self.model(x)





def training_fnn(fnn_net, epochs, inputs_net, real_out):
    '''
    Training phase for a Feed Forward Neural Network (FNN).

    :param nn.Module fnn_net: FNN model
    :param int epochs: epochs for the training phase.
    :param tensor inputs_net: matrix of inputs for the network
        with dimensions n_input x n_images.
    :param tensor real_out: tensor representing the real output
        of the network.
    '''
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(fnn_net.parameters(), lr=0.0001)
    correct = 0
    total = 0

    fnn_net = fnn_net.to(device)
    inputs_net = inputs_net.to(device)
    for i in range(len(real_out)):
        real_out[i] = real_out[i].to(device)

    final_loss = []
    batch_size = 128
    print('FNN training initialized')
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i in range(inputs_net.size()[0] // batch_size):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = fnn_net((inputs_net[i * batch_size:(i + 1) *
                                         batch_size, :]).to(device))
            loss = criterion(
                outputs,
                torch.LongTensor(real_out[i * batch_size:(i + 1) *
                                          batch_size]).to(device))
            loss.backward(retain_graph=True)
            optimizer.step()


            _, predicted = torch.max(outputs.data, 1)
            labels = torch.LongTensor(real_out[i * batch_size:(i + 1) *
                                               batch_size]).to(device)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
    print('FNN training completed', flush = True)
