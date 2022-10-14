import torch
import numpy as np
import random
from itertools import count # izip for maximum efficiency
from smithers.ml.utils import PossibleCutIdx, get_seq_model
import sys
from functools import reduce

sys.path.insert(0, '../')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class cut_off_index_estimation():

    def __init__(self, model, train_dataset, class_number, examples_per_class):
        self.model = model
        self.train_dataset = train_dataset
        self.class_number = class_number
        self.examples_per_class = examples_per_class
        self.samples_index_lists = [random.choices([i for i, j in zip(count(), train_dataset.targets) if j == h], k = self.examples_per_class) for h in range(self.class_number)]
        self.input_type = self.train_dataset.__getitem__(0)[0].dtype
        # samples_index_list is a list of class_number lists, in which the n-th contains examples_per_class indexes of elements of class n
        self.indexes_available = PossibleCutIdx(get_seq_model(self.model))


    def compute_partial_outputs(self, index):
        """
        Returns the outputs from the pre-model on the given index 
        of the samples randomly chosen.

        The outputs are expressed as a list of lists. The n-th element of the list is a list
        containing the tensorial outputs of the samples from the n-th class
        """
        pre_model = get_seq_model(self.model)[:index].to(device)
        return [[pre_model(self.train_dataset[i][0].to(device)).to(device) for i in self.samples_index_lists[j]] for j in range(self.class_number)]


    def compute_barycenters(self, outputs): #manca fattore moltiplicativo
        """
        Returns a list of class_number elements. 
        The n-th element is a tensor representing the average tensor of the outputs of the
        samples from the n-th class
        """
        sums_list = [reduce(torch.Tensor.add_, outputs[i], torch.zeros_like(outputs[i][0])) for i in range(self.class_number)]
        return list(map(lambda x: x / self.examples_per_class, sums_list))


    def subtract_barycenter(self, tensor_list, barycenter):
        """
        Returns a list of the same length of tensor_list.
        The n-th element of the returned list is the tensor obtained by subracting the tensor barycenter
        from the n-th tensor of the input list. After that the absolute value is taken component-wise.
        """
        return list(map(lambda x: torch.abs(torch.sub(x.to(device), barycenter.to(device))), tuple(tensor_list)))


    def compute_class_variances(self, outputs, barycenters):
        """
        Returns a list of class_number numbers.
        The n-th element of the list is a value describing the variance of the outputs of elements
        belonging to the n-th class
        """
        observations_variance_list = [self.subtract_barycenter(outputs[i], barycenters[i]) for i in range(self.class_number)]
        return [np.sum(np.array([torch.sum(observations_variance_list[i][j] / torch.prod(torch.tensor(observations_variance_list[i][j].shape))).detach().to('cpu').numpy() for j in range(self.examples_per_class)])) for i in range(self.class_number)]

    def return_estimated_index(self):
        """
        Returns an int.
        The int is taken as the argmin of a list containing indexes_available numbers/scores.
        Each scores represents the badness of the corresponding cut-off index.
        The semantic of the score is: high = bad, low = good.
        
        Such score is computed using the following criteria:
        1. the distances between the barycenters of the outputs of the classes should be high
        2. the variances of the outputs of the single classes should be low
        3. the higher the index, the higher the penalty
        """
        index_scores = np.zeros(len(self.indexes_available))
        #print(self.indexes_available)
        for j,i in enumerate(self.indexes_available):
            partial_outputs_tensor = self.compute_partial_outputs(i)
            barycenters = self.compute_barycenters(partial_outputs_tensor)
            barycenters_distance_matrix = torch.tensor([[torch.dist(barycenters[i], barycenters[j]) for j in range(len(barycenters))] for i in range(len(barycenters))], device=device)
            #computing distances between barycenters
            barycenters_distance_score = torch.norm(barycenters_distance_matrix)
            #computing in-class variances
            class_variances = self.compute_class_variances(partial_outputs_tensor, barycenters)
            class_variances_score = np.sum(class_variances)

            index_scores[j] = float((i+1)**(3) + -3 * barycenters_distance_score + class_variances_score)
            #print(f"layer {i}\nbarycenter distance score: {barycenters_distance_score} \nclass variance score: {class_variances_score}\n\n")
        #print(index_scores)
        #print(self.indexes_available[int(np.argmin(index_scores))]/2)
        return self.indexes_available[int(np.argmin(index_scores))]

if __name__ == '__main__':
    from smithers.ml.utils import get_seq_model
    import torch.optim as optim
    from torch import nn
    from smithers.ml.vgg import VGG

    import torch
    import sys
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    sys.path.insert(0, '../')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch_size = 8 
    data_path = '../datasets/'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root=data_path + 'CIFAR10/',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
    
    test_dataset = datasets.CIFAR10(root=data_path + 'CIFAR10/',
                                    train=False,
                                    download=True,
                                    transform=transform_test)
    #train_labels = torch.tensor(train_loader.dataset.targets).to(device)
    #targets = list(train_labels)

    VGGnet = VGG(    cfg=None,
                 classifier='cifar',
                 batch_norm=False,
                 num_classes=10,
                 init_weights=False,
                 pretrain_weights=None)
    VGGnet = VGGnet.to(device)
    VGGnet.make_layers()
    VGGnet._initialize_weights()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(VGGnet.parameters(), lr=0.001, momentum=0.9)


    def load_checkpoint(model, checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])



    pretrained = '/u/s/szanin/Smithers/smithers/ml/tutorials/check_vgg_cifar10_60_v2.pth.tar' #Stefano's
    model = VGGnet
    load_checkpoint(model, pretrained)
    seq_model = get_seq_model(model)
    model = model.to(device)

    estimator = cut_off_index_estimation(VGGnet, train_dataset, 10, 5)
    estimator.return_estimated_index()

    optimal_indexes = []
    for i in range(10):
        estimator = cut_off_index_estimation(VGGnet, train_dataset, i+1, 20)
        optimal_indexes.append(estimator.return_estimated_index())

    print(optimal_indexes)
    plt.plot(optimal_indexes)