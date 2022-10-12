'''
Utilities to perform the creation of JSON files starting from the xml files.
'''
import json
import xml.etree.ElementTree as ET
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("voc07_path", help="Path to VOC2007 folder", type=str)
parser.add_argument("voc12_path", help="Path to VOC2012 folder")
parser.add_argument("output_folder",
                    help="Path to JSON output folder",
                    type=str)
args = parser.parse_args()

voc07_path = args.voc07_path
voc12_path = args.voc12_path
output_folder = args.output_folder

# Label map
# NOTE: The labels have to be written using lower case, since in the function
# parse_annotation the label is transformed in the lower_case mode in order to
# avoid problems if in the labeling phase a label was written in a wrong way.
labels_list = ('aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
#labels_list = ('cat', 'dog')
label_map = {k: v + 1 for v, k in enumerate(labels_list)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping


def parse_annotation(annotation_path):
    '''
    :param string annotation_path: string for the path to Annotations
    return dict: dictionary containing boxes, labels, difficulties for the
        different objects in a picture
    '''
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for obj in root.iter('object'):

        difficult = int(obj.find('difficult').text == '1')
        label = obj.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)# - 1
        ymin = int(bbox.find('ymin').text)# - 1
        xmax = int(bbox.find('xmax').text)# - 1
        ymax = int(bbox.find('ymax').text)# - 1
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, voc12_path, out_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects
    in these images, and save these to file.

    :param string voc07_path: path to the 'VOC2007' folder
    :param string voc12_path: path to the 'VOC2012' folder
    :param string out_folder: folder where the JSONs must be saved
    :output json files: saved json files obtained from our dataset
        (images + xml files) saved in the output folder chosen
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)
    print(voc07_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in [voc07_path, voc12_path]:
        print(path)
        if path is not None:
            # Find IDs of images in training data
            with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
                ids = f.read().splitlines()
            for ID in ids:
                # Parse annotation's XML file
                objects = parse_annotation(
                    os.path.join(path, 'Annotations', ID + '.xml'))
                if len(objects) == 0:
                    continue
                n_objects += len(objects)
                train_objects.append(objects)
                train_images.append(os.path.join(path, 'JPEGImages', ID + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(out_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(out_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(out_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print(
        '\nThere are %d training images containing a total of %d objects.\
        Files have been saved to %s.'                                                                                                                                                                                          % (len(train_images), n_objects,\
        os.path.abspath(out_folder)))

    # Test data
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in the test data
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for ID in ids:
        # Parse annotation's XML file
        ID = ID[0:6]
        objects = parse_annotation(
            os.path.join(voc07_path, 'Annotations', ID + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', ID + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(out_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(out_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print(
        '\nThere are %d test images containing a total of %d objects.\
        Files have been saved to    %s.'                                                                                                                                                                                   % (len(test_images), n_objects,\
        os.path.abspath(out_folder)))


create_data_lists(voc07_path, voc12_path, output_folder)
