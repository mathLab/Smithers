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
from smithers.ml.models.aux_conv import AuxiliaryConvolutions
from smithers.ml.models.predictor import PredictionConvolutions
from smithers.ml.dataset.pascalvoc_dataset import PascalVOCDataset
from smithers.ml.models.detector import Detector, Reduced_Detector
from smithers.ml.models.utils import create_prior_boxes, save_checkpoint_objdet
from smithers.ml.netadapter import NetAdapter
from smithers.ml.utils import get_seq_model, Total_param, Total_flops

import warnings
warnings.filterwarnings("ignore")

text = 'STARTED'
print(f'{text:#^30}')

############################################################
############################################################


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



############################################################
############################################################


# Data parameters
data_folder = 'VOC_dog_cat/JSONfiles' #folder with json data files
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




############################################################
############################################################


#checkpoint = 'checkpoint_ssd300.pth.tar'
init_time = time()

#base_net = VGG(classifier='ssd', init_weights=False, pretrain_weights=checkpoint)
base_net = VGG(classifier='ssd', init_weights=False)
seq_model = get_seq_model(base_net)
# print(seq_model)
cutoff_idx = 7
mode_list_batch=[2, 34, 3, 3]
red_dim = mode_list_batch[-1]*mode_list_batch[-1]*mode_list_batch[1]
red_method = 'HOSVD'
inout_method = None
netadapter = NetAdapter(cutoff_idx, red_dim, red_method, inout_method)
red_model = netadapter.reduce_net(seq_model, train_dataset, None, train_loader, n_classes, device = device, mode_list_batch = mode_list_batch)
print(f'The reduced model is as follows:\n\n{red_model}')
base_net = red_model.premodel
aux_conv = red_model.proj_model
print(aux_conv)
#cfg_tot_ssd = [512,1024,] #channel number
#cfg_tot = [256, 50] #, 512, 256, 256, 256] #no hosvd
cfg_tot = [256, mode_list_batch[1]] #, 512, 256, 256, 256] #per hosvd
n_boxes = [4, 6]
predictor = PredictionConvolutions(n_classes, cfg_tot, n_boxes)
network = [base_net, aux_conv, predictor]

#create prior boxes custom for reduced net
#fmaps_dims = {'premodel': 38, 'projmodel': 1} #no hosvd
fmaps_dims = {'premodel': 38, 'projmodel': mode_list_batch[-1]}
obj_scales = {'premodel': 0.1, 'projmodel': 0.725} #0.9
aspect_ratio = {'premodel': [1., 2., 0.5], 'projmodel': [1., 2., 3., 0.5, 0.333]}
priors_cxcy = create_prior_boxes(fmaps_dims, obj_scales, aspect_ratio)
init_end = time()

#img_path = 'VOC_cat-dog/JPEGImages/000122.jpg'
#img_path = 'VOC_cat-dog/JPEGImages/002215.jpg'
img_path = '/u/s/szanin/Smithers/smithers/ml/tutorials/VOC_dog_cat/JPEGImages/001462.jpg'


original_image = Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')




############################################################
############################################################

check = None
#check = 'checkpoint_ssd300_full_trained2000.pth.tar'
epochs = 3000
start = time()
detector = Reduced_Detector(network, check, priors_cxcy, n_classes, epochs,
                    batch_size, print_freq, lr, decay_lr_at,
                    decay_lr_to, momentum, weight_decay, grad_clip,
                    train_loader, test_loader)
print(f'The detector.model is as follows:\n\n{detector.model}')

start = time()
check, loss_values, mAP_values = detector.train_detector_with_eval(label_map=label_map)
end = time()
print(f'Time needed for training: {round(end-start,2)} seconds, i.e. {round((end-start)/60,1)} minutes')


start_test = time()
check = 'checkpoint_ssd300.pth.tar'
detector.eval_detector(label_map, check)
detector.detect(original_image,
                check,
                label_map,
                min_score=0.01,
                max_overlap=0.45,
                top_k=5).show()
end_test = time()
print(f'Time needed for testing: {round(end_test-start_test,2)} seconds, i.e. {round((end_test-start_test)/60,1)} minutes')




############################################################
############################################################

np.save('mAPx10_list', np.array(mAP_values))
np.save('loss_list', np.array(loss_values))

print(f'Final loss is {loss_values[-1]}.\nFinal mAP is {mAP_values[-1]/10}.')

import matplotlib.pyplot as plt
plt.figure(figsize = [10,10])
plt.title("Loss vs mAP")
plt.xlabel("loss values")
plt.ylabel("mAP values")
plt.scatter(loss_values, mAP_values, c = np.arange(len(loss_values)), s = 20, cmap = 'gist_gray')
plt.savefig('loss vs mAP.jpg')

plt.figure(figsize = [10,10])
plt.title("mAP over epochs")
plt.xlabel("epoch")
plt.ylabel("mAP values")
plt.plot(mAP_values)
plt.savefig('mAPs.jpg')


text='COMPLETED'
print(f'{text:#^30}')
