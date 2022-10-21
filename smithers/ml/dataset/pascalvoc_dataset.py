'''
Module focused on the preparation of the dataset for the training
and testing phases for the problem of object detection using
the PascalVOC notation.
'''
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from smithers.ml.models.utils import transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create
    batches.
    """
    def __init__(self, data_folder, split, keep_difficult = False):
        """
        :param string data_folder: folder where json data files are stored
        :param string split: string that define the type of split in
            consideration, values accepted are 'TRAIN' or 'TEST'
        :param bool keep_difficult: Boolean value to determine the difficult of
            objects.  If True, objects that are considered difficult to detect
            are kept, otherwise if False they are discarded.
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'),
                  'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'),
                  'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        '''
        :param int i: integer number indicating the image we are taking into
            consideration
        :return: 4 tensors: image, boxes, labels and difficulties
        '''
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.BoolTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[~difficulties]
            labels = labels[~difficulties]
            difficulties = difficulties[~difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image,
                                                       boxes,
                                                       labels,
                                                       difficulties,
                                                       split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        '''
        :return: an integer that stand for the number of images in the
        considered split
        '''
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a
        collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We
        use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param iterable batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding
        boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties
        # tensor (N, 3, 300, 300), 3 lists of N tensors each
