'''
Class that handles the creation of a Feedforward Neural
Network (FNN).
'''

import torch
import torch.nn as nn
import torch.optim as optim


class FNN(nn.Module):
    '''
    Construction of a Feedforward Neural Network (FNN) with
    given a number of input and output neurons, one hidden
    layer with a number n_hid of neurons.

    :param int n_input: number of input neurons
    :param int n_class: number of output neurons that corresponds
        to the number of classes that compose the dataset.
    :param int n_hid: number of hidden neurons.
    '''
    def __init__(self, n_input, n_class, n_hid):
        super(FNN, self).__init__()
        self.n_input = n_input
        self.n_class = n_class
        self.n_hid = n_hid
        self.fc1 = nn.Linear(self.n_input, self.n_hid)
        self.fc2 = nn.Linear(self.n_hid, self.n_class)
        #self.fc3 = nn.Linear(self.n_hid, self.n_hid)

    def forward(self, x):
        '''
        Forward Phase.

        :param tensor x: input of the network with dimensions
            n_images x n_input
        :return: output of the FNN n_images x n_class
        :rtype: tensor
        '''
        x = torch.nn.Softplus()(self.fc1(x))
        #x = torch.nn.Softplus()(self.fc3(x))
        x = self.fc2(x)
        return x



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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fnn_net.parameters(), lr=0.001)
    correct = 0
    total = 0

    final_loss = []
    batch_size = 128
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i in range(inputs_net.size()[0] // batch_size):
        # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = fnn_net(inputs_net[i * batch_size:(i + 1) *
                                         batch_size, :])
            loss = criterion(
                outputs,
                torch.LongTensor(real_out[i * batch_size:(i + 1) *
                                          batch_size]))
            loss.backward(retain_graph=True)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 500 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 50))
                if i == inputs_net.size()[0] // batch_size - 1:
                    final_loss.append(running_loss / 50)
                running_loss = 0.0

            _, predicted = torch.max(outputs.data, 1)
            labels = torch.LongTensor(real_out[i * batch_size:(i + 1) *
                                               batch_size])
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
