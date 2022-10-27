'''
Create reduced version of SSD300
'''

import argparse
import torch
from PIL import Image
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from torch.utils import data
import pickle

from smithers.ml.vgg import VGG
from models.aux_conv import AuxiliaryConvolutions
from models.predictor import PredictionConvolutions
from dataset.pascalvoc_dataset import PascalVOCDataset
from models.detector import Detector
from models.utils import create_prior_boxes, save_checkpoint
from smithers.ml.netadapter import NetAdapter
from smithers.ml.utils import get_seq_model, Total_param, Total_flops




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

voc_labels = ('cat', 'dog')
'''voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')'''
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
n_classes = len(label_map)
print('categories:',label_map)
print('n_classes:', n_classes)

# Data parameters
#folder with json data files
data_folder = './'

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

epochs = iterations // (len(train_dataset) // 32)
decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]
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

#checkpoint = 'checkpoint_ssd300_pascal_catdog_500.pth.tar'
init_time = time()

#base_net = VGG(classifier='ssd', init_weights=False, pretrain_weights=checkpoint)
base_net = VGG(classifier='ssd', init_weights=False)
seq_model = get_seq_model(base_net)
print(seq_model)
cutoff_idx = 7
red_dim = 50
red_method = 'POD'
inout_method = 'FNN'
netadapter = NetAdapter(cutoff_idx, red_dim, red_method, inout_method)
red_model = netadapter.reduce_net(seq_model, train_dataset, None, train_loader, n_classes)
print(red_model)
base_net = red_model.premodel
aux_conv = red_model.proj_model
print(aux_conv)
cfg_tot = [256, 50] #, 512, 256, 256, 256]
n_boxes = [4, 6]
predictor = PredictionConvolutions(n_classes, cfg_tot, n_boxes)
network = [base_net, aux_conv, predictor]

#create prior boxes custom for reduced net
fmaps_dims = {'premodel': 38, 'projmodel': 1}
obj_scales = {'premodel': 0.1, 'projmodel': 0.725} #0.9
aspect_ratio = {'premodel': [1., 2., 0.5], 'projmodel': [1., 2., 3., 0.5, 0.333]}
priors_cxcy = create_prior_boxes(fmaps_dims, obj_scales, aspect_ratio)
init_end = time()
print('time needed to initialize the model', init_end - init_time)

#img_path = 'voc_dir/VOC_cat-dog/JPEGImages/000122.jpg'
#img_path = 'voc_dir/VOC_cat-dog/JPEGImages/002215.jpg'
img_path = 'VOCdevkit/VOC2007/JPEGImages/001424.jpg'


original_image = Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')


check = None
epochs = 500
start = time()
print(epochs)
detector = Detector(network, check, priors_cxcy, n_classes, epochs,
                    batch_size, print_freq, lr, decay_lr_at,
                    decay_lr_to, momentum, weight_decay, grad_clip,
                    train_loader, test_loader)
#check = save_checkpoint(0, network, None)
print(detector.model)
'''
check = torch.load(check)
model = check['model']

rednet_storage = torch.zeros(4)
rednet_flops = torch.zeros(4)

rednet_storage[0], rednet_storage[1], rednet_storage[2], rednet_storage[3] = [
       Total_param(model[0]),
       Total_param(model[1]),
       Total_param(model[2].features_loc),
       Total_param(model[2].features_cl)]


print('SSD300 reduced-storage_init')
print(
      ' Pre nnz = {:.2f}, POD_model nnz={:.2f}, feature_loc nnz={:.4f}, feature_cl nnz={:.4f}'.format(
                  rednet_storage[0], rednet_storage[1],
                  rednet_storage[2], rednet_storage[3]))
'''

start = time()
#check, loss_value = detector.train_detector()
end = time()
print('tìme needed for train and test', end-start)


start_test = time()
#check = 'checkpoint_ssd300.pth.tar'
detector.eval_detector(label_map, check)
detector.detect(original_image,
                check,
                label_map,
                min_score=0.01,
                max_overlap=0.45,
                top_k=5).show()
end_test = time()
print('Time needed to test the detector', end_test-start_test)

#check = 'checkpoint_ssd300_red_pascalvoc.pth.tar'
#check = 'checkpoint_ssd300.pth.tar'
check = torch.load(check)
model = check['model']

rednet_storage = torch.zeros(4)
rednet_flops = torch.zeros(4)

rednet_storage[0], rednet_storage[1], rednet_storage[2], rednet_storage[3] = [
       Total_param(model[0]),
       Total_param(model[1]),
       Total_param(model[2].features_loc),
       Total_param(model[2].features_cl)]


#rednet_flops[0], rednet_flops[1], rednet_flops[2], rednet_flops[3] = [
#        Total_flops(model[0], device),
#        Total_flops(model[1], device),
#        Total_flops(model[2].features_loc, device),
#        Total_flops(model[2].features_cl, device)]

print('SSD300 reduced-storage')
print(
      ' Pre nnz = {:.2f}, POD_model nnz={:.2f}, feature_loc nnz={:.4f}, feature_cl nnz={:.4f}'.format(
                  rednet_storage[0], rednet_storage[1],
                  rednet_storage[2], rednet_storage[3]))

#        print(
#              '   flops:  Pre = {:.2f}, POD_model = {:.2f}, ANN ={:.2f}'.format(
#                  rednet_flops[0], rednet_flops[1], rednet_flops[2]))


#check1 = 'checkpoint_ssd300_pascal_catdog_500.pth.tar'
check1 = 'checkpoint_ssd300_catdog_500_new.pth.tar'
check1 = torch.load(check1)
model = check1['model']

rednet_storage = torch.zeros(4)
rednet_flops = torch.zeros(4)

rednet_storage[0], rednet_storage[1], rednet_storage[2], rednet_storage[3]  = [
       Total_param(model[0]),
       Total_param(model[1].features),
       Total_param(model[2].features_loc),
       Total_param(model[2].features_cl)]

rednet_vgg_storage = torch.zeros(4)
rednet_vgg_storage[0], rednet_vgg_storage[1], rednet_vgg_storage[2], rednet_vgg_storage[3],  = [
       Total_param(model[0]),
       Total_param(model[0].features),
       Total_param(model[0].avgpool),
       Total_param(model[0].classifier)]

#rednet_flops[0], rednet_flops[1], rednet_flops[2], rednet_flops[3] = [
#        Total_flops(model[0], device),
#        Total_flops(model[1], device),
#        Total_flops(model[2].features_loc, device),
#        Total_flops(model[2].features_cl, device)]


print('SSD300-storage')
print(
      ' Pre nnz = {:.2f}, aux_model nnz={:.2f}, feature_loc nnz={:.4f}, feature_cl nnz={:.4f}'.format(
                  rednet_storage[0], rednet_storage[1],
                  rednet_storage[2], rednet_storage[3]))

print(
      ' Pre nnz = {:.2f}, pre_vgg nnz={:.2f}, pre_avgpool nnz={:.4f}, pre_classifier nnz={:.4f}'.format(
                  rednet_vgg_storage[0], rednet_vgg_storage[1],
                  rednet_vgg_storage[2], rednet_vgg_storage[3]))


torch.save(detector, 'check_ssd300_red.pth')
