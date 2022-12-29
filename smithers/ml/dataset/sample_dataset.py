'''
Utilities to extract datasets of N images divided in M classes
from a whole dataset.
'''

import os
import shutil


from xml.dom import minidom


def select_indeces(valid_classes, max_imgs, xmlfolder):
    '''
    Function that selects the indexes of the images of a
    specific class contained in the source dataset.

    :param list(str) valid_classes: list of strings defining
        the selected categories.
    :param int max_imgs: maximum number of images to consider.
    :param str xmlfolder: relative path to the folder
        containing the annotations (xml files).
    :return: valid_ids, list containg the selected indexes.
    :rtype: list
    '''
    valid_ids = []
    file = 'dataset.txt'
    out_file = open(file, 'w')

    for id_ in range(1, 10000):
        id_ = '00{:04d}'.format(id_)
        xml_ = os.path.join(xmlfolder, '{}.xml'.format(id_))
        file_ = minidom.parse(xml_)
        classes = file_.getElementsByTagName('object')
        classes = [class_.getElementsByTagName('name')[0] for class_ in classes]
        classes = [class_.firstChild.nodeValue for class_ in classes]

        if all([class_ in valid_classes for class_ in classes]):
            valid_ids.append(id_)
            out_file.write(id_ + '\n')

        if len(valid_ids) >= max_imgs:
            break
    out_file.close()
    return valid_ids

def copy_imgs(ids, src, dst, folder='JPEGImages'):
    '''
    Function copying selected images from a directory.
    '''

    os.mkdir(os.path.join(dst, folder))

    for id_ in ids:
        src_ = os.path.join(src, folder, '{}.jpg'.format(id_))
        dst_ = os.path.join(dst, folder, '{}.jpg'.format(id_))
        shutil.copyfile(src_, dst_)

def copy_xmls(ids, src, dst, folder='Annotations'):
    '''
    Function copying selected xml files from a directory.
    '''

    os.mkdir(os.path.join(dst, folder))

    for id_ in ids:
        src_ = os.path.join(src, folder, '{}.xml'.format(id_))
        dst_ = os.path.join(dst, folder, '{}.xml'.format(id_))
        shutil.copyfile(src_, dst_)

def sample_dataset(src_dataset, dst_dataset):
    '''
    Function that performs the sampling of the dataset.

    :param str src_dataset: path to the source dataset.
    :param str dst_dataset: path to the output dataset.
    '''
    os.mkdir(dst_dataset)

    ids = select_indeces(['dog', 'cat'], 300,
                         os.path.join(src_dataset, 'Annotations'))

    copy_imgs(ids, src_dataset, dst_dataset)
    copy_xmls(ids, src_dataset, dst_dataset)



if __name__ == '__main__':

    sample_dataset('../VOCdevkit/VOC2007', 'VOC_dog_cat')
