'''
Module focused on the creation of a custom dataset class in order
to use our custom dataset for the problem of image recognition
and thus classification.
'''
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# CUSTOM DATASET CLASS
class Imagerec_Dataset(Dataset):
    '''
    Class that handles the creation of a custom dataset class to
    be used by data loader.
    :param pandas.DataFrame img_data: tabular containing all the
        relations (image, label)
    :param str img_path: path to the directiory containing all the
        images
    :param transform_obj transform: list of transoforms to apply to
        images. Defaul value set to None.
    :param list resize_dim: list of integers corresponding to the
        size to which we want to resize the images
    '''
    def __init__(self, img_data, img_path, resize_dim, transform=None):
        self.img_data = img_data
        self.img_path = img_path
        self.resize_dim = resize_dim
        self.transform = transform
        self.targets = self.img_data['encoded_labels']

    def __len__(self):
        '''
	Function that returns the number of images in the dataset
        :return int: integer number representing the number of
            images in the dataset
        '''
        return len(self.img_data)

    def __getitem__(self, index):
        '''
	Function that returns the data and labels
        :param int index: number representing a specific image in the
            dataset
        :return tensor image, label: image and label associated
            with the index given as input
        '''
        img_name = os.path.join(self.img_path,
                                self.img_data.loc[index, 'labels'],
                                self.img_data.loc[index, 'Images'])
        image = Image.open(img_name)
        image = image.resize((self.resize_dim[0], self.resize_dim[1]))
        label = torch.tensor(self.img_data.loc[index, 'encoded_labels'])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label

    def getdata(self, index):
        '''
	Function that returns a subset of the dataset
        :param list index: number representing a specific image in the
            dataset
        :return: subset of the dataset composed by obs of type (img, label)
        :rtype: list
        '''
        output = []
        for idx in index:
            image, label = self.__getitem__(idx)
            output.append([image, label])
        return output
