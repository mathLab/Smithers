'''
Train SSD300 on PascalVOC dataset (VOC2007)
'''

import torch
from PIL import Image
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms

from .vgg import VGG
from models.aux_conv import AuxiliaryConvolutions
from models.predictor import PredictionConvolutions
from dataset.pascalvoc_dataset import PascalVOCDataset
from models.detector import Detector
from models.utils import create_prior_boxes

# Learning parameters
batch_size = 8  # batch size
workers = 4  # number of workers for loading data in the DataLoader
iterations = 120000  # number of iterations to train
print_freq = 200  # print training status every __ batches
lr = 1e-4  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1
# decay learning rate to this fraction of the existing learning rate
#n_classes = 6
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None
# clip if gradients are exploding, which may happen at larger batch sizes

#voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat',
#        'bottle', 'bus', 'car', 'cat', 'chair',
#        'cow', 'diningtable', 'dog', 'horse',
#        'motorbike', 'person', 'pottedplant',
#        'sheep', 'sofa', 'train', 'tvmonitor')
voc_labels = ('cat', 'dog')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
n_classes = len(label_map)
print('categories:',label_map)
print('n_classes:', n_classes)

# Data parameters
data_folder = 'voc_dir/VOC_cat-dog/JSONfiles' #folder with json data files
keep_difficult = True


train_dataset = PascalVOCDataset(data_folder,
                                 split='train',
                                 keep_difficult=keep_difficult)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=train_dataset.collate_fn,
    num_workers=workers,
    pin_memory=True)

epochs = iterations // (len(train_dataset) // 16) #500
decay_lr_at = [it // (len(train_dataset) // 16) for it in decay_lr_at]
print('Training images:', len(train_dataset))
# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          collate_fn=test_dataset.collate_fn,
                                          num_workers=workers,
                                          pin_memory=True)
print('Testing images:', len(test_dataset))

start_init = time()
base_net = VGG(classifier='ssd', init_weights=False)
aux_conv = AuxiliaryConvolutions()
predictor = PredictionConvolutions(n_classes)
network = [base_net, aux_conv, predictor]
priors_cxcy = create_prior_boxes()
end_init = time()
print('Time needed to initialize the net', end_init-start_init)

check = None 
#check = 'checkpoint_ssd300_tutorial.pth.tar'
epochs = 500
start = time()
detector = Detector(network, check, priors_cxcy, n_classes, epochs,
                        batch_size, print_freq, lr, decay_lr_at,
                        decay_lr_to, momentum, weight_decay, grad_clip,
                        train_loader, test_loader)
print(detector.model)
check, loss_value = detector.train_detector()
end = time()
print('tìme needed for train and test', end-start)
start_test = time()
epo = np.arange(start=0, stop=epochs, step=1)
plt.plot(epo, loss_value)
plt.xlabel('Epochs')
plt.ylabel('Value Loss')
plt.savefig('loss_pascal_cat.png')
detector.eval_detector(label_map, check)
end_test = time()

print('tìme needed for train and test', end_test-start_test)


img_path = 'voc_dir/VOC2007/JPEGImages/002215.jpg'
#img_path = 'voc_dir/VOC_cat-dog/JPEGImages/000122.jpg'
original_image = Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')

detector.detect(original_image,
                check,
                label_map,
                min_score=0.01,
                max_overlap=0.45,
                top_k=5).show()
