'''
Class that handles the creation of an ANN for the last
part of our reduced net.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,  n_class, n1):
        super(Net, self).__init__()
        self.n_class = n_class
        self.fc1 = nn.Linear(50, n1)
        self.fc2 = nn.Linear(n1, self.n_class)
#        self.fc3 = nn.Linear(n1, n1)
#        self.fc4 = nn.Linear(n1, n1)
#        self.fc5 = nn.Linear(n1, n1)
#        self.fc6 = nn.Linear(10, 10)

    def forward(self, x):
#        x = torch.tanh(self.fc1(x))
        x = torch.nn.Softplus()(self.fc1(x))
#        x = F.relu(self.fc1(x))
#        x = torch.tanh(self.fc3(x))
#        x = torch.nn.Softplus()(self.fc3(x))
#        x = torch.nn.Softplus()(self.fc4(x))
#        x = torch.nn.Softplus()(self.fc5(x))
#        x = torch.nn.Softplus()(self.fc6(x))
        x = self.fc2(x)
        return x


    def training(self, epochs, inputs_net, real_out):
    '''
    Training phase for a Feed Forward Neural Network (FNN)
    :param
    :param
    :param
    :param
    '''
    # TRAINING NET
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        correct = 0
        total = 0

        final_loss = []
        batch_size = 128
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i in range(inputs_net.size()[0]// batch_size):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs_net[i*batch_size : (i+1)*batch_size, :])
                loss = criterion(outputs, torch.LongTensor(real_out[i*batch_size : (i+1)*batch_size]))
                loss.backward(retain_graph=True)
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:    # print every 500 mini-batches
                    print('[%d, %5d] loss: %.6f' %
                         (epoch + 1, i + 1, running_loss / 50))
                    if i == inputs_net.size()[0]// batch_size -1:
                        final_loss.append(running_loss / 50)
                    running_loss = 0.0

                _, predicted = torch.max(outputs.data, 1)
                labels = torch.LongTensor(real_out[i*batch_size : (i+1)*batch_size])
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
