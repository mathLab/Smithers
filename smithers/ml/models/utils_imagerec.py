'''
Utilities for initiliazing, training and testing a network
for the problem of Image Recognition.
'''
import copy
import torch
import torch.nn as nn
import torch.optim as optim


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

def train(net, num_epochs, train_loader, test_loader, optim_str, device):
    '''
    Function performing the training of a network.

    :param nn.Module net: network under consideration
    :param int num_epochs: number of training epochs
    :param iterable train_loader: iterable object, it load the dataset for
            training. It iterates over the given dataset, obtained combining a
            dataset(images, labels) and a sampler.
    :param iterable test_loader: iterable object, it load the dataset for
            testing. It iterates over the given dataset, obtained combining a
            dataset(images, labels) and a sampler.
    :param str optim_str: optimizer to use in the training
    :param torch.device device: object representing the device on
            which a torch.Tensor is or will be allocated.
    '''
    criterion = nn.CrossEntropyLoss()
    if optim_str == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    elif optim_str == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=0.001)

    net.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            batch_size = images.size()[0]
            images = images.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = net(images)
            # compute the loss based on model output and real labels
            loss = criterion(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics at the end of the epoch
            running_loss += loss.item()     # extract the loss value
            if i == len(train_loader)-1:
                print('Epoch {}, Loss Value: {:.5f}'.format
                      (epoch + 1, running_loss / ((i+1)*batch_size)))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when
        # tested over all 10000 test images
        accuracy_train = testAccuracy(net, train_loader, device)
        print('For epoch {} the train accuracy over the whole train set' +
              'is {:.2f}%'.format(epoch + 1, accuracy_train))
        accuracy = testAccuracy(net, test_loader, device)
        print('For epoch {} the test accuracy over the whole test set' +
              'is {:.2f}%'.format(epoch + 1, accuracy))

    torch.save(copy.deepcopy(net), 'check_network.pth')

def testAccuracy(net, test_loader, device):
    '''
    Function for testing the accuracy of the model.

    :param nn.Module net: network under consideration
    :param iterable test_loader: iterable object, it load the dataset for
            testing. It iterates over the given dataset, obtained combining a
            dataset(images, labels) and a sampler.
    :param torch.device device: object representing the device on
            which a torch.Tensor is or will be allocated.
    '''
    net.eval()
    accuracy = 0.0
    total = 0.0
    net.to(device)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # run the model on the test set to predict labels
            outputs = net(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy


def testClasses(net, n_classes, test_loader, classes, device):
    '''
    Function testing the accuracy reached for each class
    composing the dataset.

    :param nn.Module net: network under consideration
    :param int n_classes: number of classes composing the dataset
    :param iterable test_loader: iterable object, it load the dataset for
            testing. It iterates over the given dataset, obtained combining a
            dataset(images, labels) and a sampler.
    :param torch.device device: object representing the device on
            which a torch.Tensor is or will be allocated.
    '''
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    net.eval()
    net.to(device)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            if len(labels) == 1:
                c = torch.tensor([c])
            for i, _ in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(n_classes):
        print('Accuracy of {} : {:.2f}%'.format(
            classes[i], 100 * class_correct[i] / class_total[i]))
