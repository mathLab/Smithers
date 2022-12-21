'''
Utilities for the transformations needed inside a CNN and performing
object detection
'''
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


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


def find_intersection(set_1, set_2):
    '''
    Find the intersection of every box combination between two sets of boxes
    that are in boundary coordinates.

    :param torch.Tensor set_1: set 1, a tensor of dimensions (n1, 4)
    :param torch.Tensor set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each
        of the boxes in set 2, i.e. a tensor of dimensions (n1, n2)
    :rtype: torch.Tensor
    '''

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1),
                             set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1),
                             set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds,
                                    min=0)  # (n1, n2, 2)
    # intersection is given by the area of the box, those the area of
    # a rectangular
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    '''
    Find the Jaccard Overlap (IoU) of every box combination between two
    sets of boxes that are in boundary coordinates.

    :param torch.Tensor set_1: set 1, a tensor of dimensions (n1, 4)
    :param torch.Tensor set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap (float) of each of the boxes in set 1 with
         respect to each of the boxes in set 2, i.e. a tensor of
         dimensions (n1, n2)
    :rtype: torch.Tensor
    '''

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1]
                                                 )  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1]
                                                 )  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(
        0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def xy_to_cxcy(xy):
    '''
    Convert bounding boxes from boundary coordinates
    (x_min, y_min, x_max, y_max) to center-size
    coordinates (c_x, c_y, w, h).

    :param torch.Tensor xy: bounding boxes in boundary coordinates, a tensor
        of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of
        size (n_boxes, 4)
    :rtype: torch.Tensor
    '''
    return torch.cat(
        [
            (xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
            xy[:, 2:] - xy[:, :2]
        ],
        1)  # w, h


def cxcy_to_xy(cxcy):
    '''
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h)
    to boundary coordinates (x_min, y_min, x_max, y_max).

    :param torch.Tensor cxcy: bounding boxes in center-size coordinates, a
        tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of
        size (n_boxes, 4)
    :rtype: torch.Tensor
    '''
    return torch.cat(
        [
            cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
            cxcy[:, :2] + (cxcy[:, 2:] / 2)
        ],
        1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    '''
    Encode bounding boxes (that are in center-size form) w.r.t.
    the corresponding prior boxes (that are in center-size form).
    Thus the offests are found, i.e. how much the prior has to
    be adjusted to produce the bounding box.
    For the center coordinates, find the offset with respect to
    the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box,
    and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this
    encoded form.

    :param torch.Tensor cxcy: bounding boxes in center-size coordinates,
        a tensor of size (n_priors, 4)
    :param troch.Tensor priors_cxcy: prior boxes with respect to which
        the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    :rtype: torch.Tensor
    '''

    # The 10 and 5 below are referred to as 'variances' in the
    # original Caffe repo, completely empirical. They are for some
    # sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat(
        [
            (cxcy[:, :2] - priors_cxcy[:, :2]) /
            (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
            torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5
        ],
        1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    '''
    Decode bounding box coordinates predicted by the model, since
    they are encoded in the form mentioned above.
    (see function cxcy_to_gcxgcy)
    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param torch.Tensor gcxgcy: encoded bounding boxes, i.e. output of the
        model, a tensor of size (n_priors, 4)
    :param torch.Tensor priors_cxcy: prior boxes with respect to which the
        encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of
        size (n_priors, 4)
    :rtype: torch.Tensor
    '''

    return torch.cat(
        [
            gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 +
            priors_cxcy[:, :2],  # c_x, c_y
            torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]
        ],
        1)  # w, h


def create_prior_boxes(fmap_dims=None, obj_scales=None, aspect_ratios=None):
    '''
    Create the 8732 prior (default) boxes for the SSD300, as defined
    in the paper.

    :param dict fmap_dict: If None, it corresponds to the dimension of
        the low and high(auxiliary) feature maps for SSD300. Otherwise,
        you need to provide a dictionary, where the keys are the
        convolutional layers and the values associated are the
        dimensions of those feature maps.
    :param dict obj_scales: If None, it returns the scaling values for
        the priors in SSD300, i.e the percentage of the image that is
        detected. In particular, in SSD300 larger feature maps as
        conv4_3 have smaller object scales, thus they will used to
        detect smaller objects. Otherwise you need to provide a
        dictionary where the keys are the convolutional layers
        corresponding to the low and high level features anf the values
        associated the scales used for each of them.
    :param dict aspect_ratios: If None, the aspect ratios used are that
        of SSD300. Otherwise, you need to provide a dictionary where
        the keys are the low and high level feature maps and the value
        associated is a list with the different aspect ratios used for
        the priors in that layer.
    :return: prior boxes in center-size coordinates, a tensor of
        dimensions (8732, 4)
    :rtype: torch.Tensor
    '''

    if fmap_dims is None:
        fmap_dims = {
            'conv4_3': 38,
            'conv7': 19,
            'conv8_2': 10,
            'conv9_2': 5,
            'conv10_2': 3,
            'conv11_2': 1
        }
    if obj_scales is None:
        obj_scales = {
            'conv4_3': 0.1,
            'conv7': 0.2,
            'conv8_2': 0.375,
            'conv9_2': 0.55,
            'conv10_2': 0.725,
            'conv11_2': 0.9
        }
    if aspect_ratios is None:
        aspect_ratios = {
            'conv4_3': [1., 2., 0.5],
            'conv7': [1., 2., 3., 0.5, .333],
            'conv8_2': [1., 2., 3., 0.5, .333],
            'conv9_2': [1., 2., 3., 0.5, .333],
            'conv10_2': [1., 2., 0.5],
            'conv11_2': [1., 2., 0.5]
        }

    fmaps = list(fmap_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([
                        cx, cy, obj_scales[fmap] * np.sqrt(ratio),
                        obj_scales[fmap] / np.sqrt(ratio)
                    ])

                    # For an aspect ratio of 1, use an additional prior
                    # whose scale is the geometric mean of the scale of
                    # the current feature map and the scale of the next
                    # feature map
                    if ratio == 1.:
                        try:
                            additional_scale = np.sqrt(obj_scales[fmap] *
                                                       obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next"
                        # feature map
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append(
                            [cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
    prior_boxes.clamp_(0, 1)  # (8732, 4)

    return prior_boxes


def expand(image, boxes, filler):
    '''
    Perform a zooming out operation by placing the image in a larger canvas
    of filler material. This helps to learn to detect smaller objects.

    :param torch.Tensor image: image, a tensor of dimensions
        (3, original_h, original_w)
    :param torch.Tensor boxes: bounding boxes in boundary coordinates, a tensor
        of dimensions (n_objects, 4)
    :param list filler: RBG values of the filler material, a list like [R, G, B]
	     where R, G, B are scalar values
    :return: expanded image, updated bounding box coordinates
    :rtype: torch.Tensor, torch.Tensor
    '''
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    # The zoomed out image must be between 1 and 4 times as large as the
    # original.
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones(
        (3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(
            1)  # (3, new_h, new_w)
    # Note - do not use expand() like
    # new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one
    # pixel will change all

    # Place the original image at random coordinates in this new image
    # (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    '''
    Performs a random crop(zoom in) in the manner stated in the SSD paper.
    Helps to learn to detect larger and partial objects. Note that some
    objects may be cut out entirely. Adapted from:
   https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param torch.Tensor image: image, a tensor of dimensions
         (3, original_h, original_w)
    :param torch.Tensor boxes: bounding boxes in boundary coordinates, a tensor
         of dimensions (n_objects, 4)
    :param torch.Tensor labels: labels of objects, a tensor of dimensions
         (n_objects)
    :param torch.Tensor difficulties: difficulties of detection of these objs,
        a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels,
        updated difficulties
    :rtype: torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    '''
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9,
                                     None])  # 'None' refers to no cropping
        #print('min_overlap', min_overlap)
        # If not cropping, thus if we have None, we exit and return the
        # original image
        if min_overlap is None:
            #print('min overlap is None')
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in
        # paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            #print('current trial', _)
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the
            # authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)
            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                #print('Aspect ratio not in the range allowed')
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(
                crop.unsqueeze(0), boxes
            )  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than
            # the minimum, try again
            if overlap.max().item() < min_overlap:
                #print(
                #    'no bounding box has Jaccard overlap greater than\
                #     the minimum'
                #)
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop, thus at the end
            # not all the bounding boxes may be in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (
                bb_centers[:, 0] < right) * (bb_centers[:, 1] >
                                             top) * (bb_centers[:, 1] < bottom)
            # size centers_in_crop: (n_objects), a Torch uInt8/Byte tensor, can
            # be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                #print('No bounding box has its center in the crop')
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2],
                                         crop[:2])  # crop[:2] is [left, top]
            # adjust to crop (by substracting crop's left,top), idem below
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:,
                      2:] = torch.min(new_boxes[:, 2:],
                                      crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]
            #print(new_image.size())
            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    '''
    Flip image horizontally.

    :param PIL Image image: image, a PIL Image
    :param torch.Tensor boxes: bounding boxes in boundary coordinates, a tensor
        of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    :rtype: PIL Image, torch.Tensor
    '''
    # Flip image horizontally
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def photometric_distort(image):
    '''
    Distort brightness, contrast, saturation, and hue, each with a 50% chance,
    in random order.

    :param PIL Image image: image, a PIL Image
    :return: distorted image
    :rtype: PIL Image
    '''
    new_image = image

    distortions = [
        FT.adjust_brightness, FT.adjust_contrast, FT.adjust_saturation,
        FT.adjust_hue
    ]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255
                #because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5
                #for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    '''
    Resize image. For the SSD300, it resizes to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes
    (w.r.t image dimensions) in this process, you may choose to retain them.

    :param PIL Image image: image, a PIL Image
    :param torch.Tensor boxes: bounding boxes in boundary coordinates, a tensor
        of dimensions (n_objects, 4)
    :param tuple dims: if None, the image is rescaled to (300,300) as needed
        for SSD300. Otherwise, a tuple with the desired dimensions need to
        be provided.
    :param bool return_percent_coords: If True, it returns the
        percent/fractional coordinates. Otherwise, it returns the new
        coordinates of the bounding boxes for the rescaled image.
    :return: resized image, updated tensor for bounding box
        coordinates.(or fractional coordinates, in which case they
        remain the same)
    :rtype: PIL Image, torch.Tensor
    '''
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor(
        [image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1],
                                      dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def transform(image, boxes, labels, difficulties, split):
    '''
    Apply the transformations above.
    NOTE: Since random_crop is used, the output size from the bounding boxes
    may differ from that of the input bounding boxes

    :param PIL Image image: image, a PIL Image
    :param torch.Tensor boxes: bounding boxes in boundary coordinates, a
        tensor of dimensions (n_objects, 4)
    :param torch.Tensor labels: a tensor of dimensions (n_objects) containing
        the labels of objects.
    :param torch.Tensor difficulties: a tensor of dimensions (n_objects)
        representing the difficulties of detection for the objects in
        the picture.
    :param str split: Expected strings are: 'TRAIN' or 'TEST', since
        different sets of transformations are applied. If a different string
        is given, an error arise.
    :return: transformed image, transformed bounding box coordinates,
        transformed labels, transformed difficulties
    :rtype: torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    '''
    assert split in {'TRAIN', 'TEST'}

    # Mean and standard deviation of ImageNet data that our base VGG from
    # torchvision was trained on.
    # See: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations if evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with
        # 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training
        # detection of small objects. Fill surrounding space with the mean
        # of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(
            new_image, new_boxes, new_labels, new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary
    #coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our
    #base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


class AverageMeter(object):
    '''
    Keeps track of most recent value, average, sum, and count of a
    metric.
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        '''
        Reset value, average, sum and count of the metric.
        '''
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        '''
        Update value, sum, counter and average of the metric.

        :param int val: current value of the object we are considering
        :param int n: integer number corresponding to the step at which
            we are updating the counter and other values
        '''
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    '''
    Clips gradients computed during backpropagation in order to avoid
    explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    '''
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint_objdet(epoch, model, optimizer, cut_idx = None, with_epochs = None):
    '''
    Save model checkpoint.

    :param scalar epoch: epoch number
    :param list model: list of constructed classes that compose our network
    :param torch.Tensor optimizer: optimizer chosen
    :return: path to the checkpoint file
    :rtype: str
    '''
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
    if cut_idx == None and with_epochs == None:
        filename = 'checkpoint_ssd300.pth.tar'
    elif cut_idx == None and with_epochs != None:
        filename = f'checkpoint_ssd300_epoch_{epoch}.pth.tar'
        torch.save(state, filename)
        filename = 'checkpoint_ssd300.pth.tar'
    else:
        filename = 'checkpoint_VGG16_ASNet_sparse_cutID_%d.pth.tar' % (cut_idx)
    torch.save(state, filename)
    return filename

def save_checkpoint_objdet_name(epoch, model, optimizer, path):
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
    torch.save(state, path)
    return path


def adjust_learning_rate(optimizer, scale):
    '''
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param float scale: factor to multiply learning rate with.
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale
    print("DECAYING learning rate.\n The new LR is %f\n" %
          (optimizer.param_groups[-1]['lr'], ))

    #NOTE: HERE INITIALLY WAS optimizer.param_groups[1]['lr'] (now -1)


def non_max_sup(n_above_min_score, class_decoded_locs, overlap, max_overlap):
    '''
    Non-Maximum Suppression (NMS)

    :param n_above_min_score
    :param torch.Tensor class_decoded_locs: (n_qualified, 4)
    :param torch.Tensor overlap: tensor containing Jaccard overlap between
        predicted boxes, size (n_above_min_score, n_above_min_score)
    :param float max_overlap: maximum overlap two boxes can have so that the
        one with the lower score is not suppressed via NMS
    :return: NMS
    :rtype: torch.Tensor
    '''
    # A torch.uint8 (byte) tensor to keep track of which predicted boxes
    # to suppress: 1 implies suppress, 0 implies don't suppress
    suppress = torch.zeros((n_above_min_score),
                           dtype=torch.uint8).to(device)  # (n_qualified)

    # Consider each box in order of decreasing scores
    for box in range(class_decoded_locs.size(0)):
        # If this box is already marked for suppression
        if suppress[box] == 1:
            continue

        # Suppress boxes whose overlaps (with this box) are greater than
        # maximum overlap (but we are keeping the box we are considering
        # in the for loop). Find such boxes and update suppress indices.
        suppress = torch.max(
            suppress,
            (overlap[box] > max_overlap).byte())  #type(torch.ByteTensor))
        # The max operation retains previously suppressed boxes, like an 'OR'
        # operation

        # Don't suppress this box, even though it has an overlap of 1 with
        # itself
        suppress[box] = 0
    return suppress


def detect_objects(priors_cxcy, predicted_locs, predicted_scores, n_classes,
                   min_score, max_overlap, top_k):
    '''
    Decipher the 8732 locations and class scores (output of the SSD300)
    to detect objects. For each class, perform Non-Maximum Suppression (NMS)
    on boxes that are above a minimum threshold.

    :param torch.Tensor priors_cxcy: priors (default bounding boxes) in
        center-size coordinates, a tensor of size (n_boxes, 4)
    :param torch.Tensor predicted_locs: predicted locations/boxes w.r.t the 8732
        prior boxes, a tensor of dimensions (N, 8732, 4)
    :param torch.Tensor predicted_scores: class scores for each of the encoded
        locations/boxes, a tensor of dimensions (N, 8732, n_classes)
    :param scalar n_classes: number of different type of objects in your dataset
    :param float min_score: minimum threshold for a box to be considered a match
        for a certain class
    :param float max_overlap: maximum overlap two boxes can have so that the one
        with the lower score is not suppressed via NMS
    :param int top_k: if there are a lot of resulting detection across all
        classes, keep only the top 'k'
    :return: detections (boxes, labels, and scores), lists of length
        batch_size
    :rtype: list, list, list
    '''
    batch_size = predicted_locs.size(0)
    n_priors = priors_cxcy.size(0)
    predicted_scores = F.softmax(predicted_scores,
                                 dim=2)  # (N, 8732, n_classes)

    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted
        # boxes to
        decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i],
                                                 priors_cxcy))
        # (8732, 4), these are fractional pt. coordinates

        # Lists to store boxes and scores for this image
        image_boxes = list()
        image_labels = list()
        image_scores = list()

        max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

        # Check for each class
        # Note: we do not consider class 0, that is associated with background.
        for c in range(1, n_classes):
            # Keep only predicted boxes and scores where scores for this class
            # are above the minimum score
            class_scores = predicted_scores[i][:, c]  # (8732)
            score_above_min_score = class_scores > min_score
            # torch.uint8 (byte) tensor, for indexing
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores = class_scores[score_above_min_score]  # (n_qualified)
            #n_qualified=n_above_min_score <= 8732
            class_decoded_locs = decoded_locs[
                score_above_min_score]  # (n_qualified, 4)

            # Sort predicted boxes and scores by scores
            class_scores, sort_ind = class_scores.sort(
                dim=0, descending=True)  # (n_qualified), (n_above_min_score)
            class_decoded_locs = class_decoded_locs[
                sort_ind]  # (n_above_min_score, 4)

            # Find the overlap between predicted boxes
            overlap = find_jaccard_overlap(
                class_decoded_locs,
                class_decoded_locs)  # (n_above_min_score, n_above_min_score)

            #################################
            # Non-Maximum Suppression (NMS) #
            #################################
            suppress = non_max_sup(n_above_min_score, class_decoded_locs,
                                   overlap, max_overlap)

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[~suppress.bool()])
            image_labels.append(
                torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
            image_scores.append(class_scores[~suppress.bool()])

        # If no object in any class is found, store a placeholder for
        # 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            image_labels.append(torch.LongTensor([0]).to(device))
            image_scores.append(torch.FloatTensor([0.]).to(device))

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        # Keep only the top k objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]  # (top_k)
            image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
            image_labels = image_labels[sort_ind][:top_k]  # (top_k)

        # Append to lists that store predicted boxes and scores for all images
        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores


def calculate_AP(true_images, true_boxes, true_difficulties, true_labels,
                 det_images, det_boxes, det_scores, det_labels, n_classes):
    '''
    Calculate the Average Precision (AP) of detected objects.

    :param torch.Tensor true_images: tensor for storing the images with the
        actual objects contained in them, size (n_objects)
    :param torch.Tensor true_boxes: tensor for storing the true boxes for the
        actual objects in our images, size (n_objects, 4)
    :param torch.ByteTensor true_difficulties: tensor for storing the true
        difficulties for the actual objects in our images, size (n_objects)
    :param torch.Tensor true_labels: tensor containing actual objects' labels
        for our images, size (n_objects)
    :param torch.Tensor det_images: tensor for storing the images containing the
        detected objects, size (n_detections)
    :param torch.Tensor det_boxes: tensor for storing the boxes of the detected
        objects in our images, size (n_detections, 4)
    :param torch.Tensor det_scores: tensor for storing the scores of the
        detected objects in our images, size (n_detections)
    :param torch.Tensor det_labels: tensor containing detected objects' labels
        in our images, size (n_objects)
    :param int n_classes: number of classes in the dataset
    :return: average_precisions (AP) corresponding to
        the mean of the recalls above the threshold chosen for each class in the
        dataset
    :rtype: torch.Tensor
    '''
    average_precisions = torch.zeros((n_classes - 1),
                                     dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels ==
                                                    c]  # (n_class_objects)
        n_easy_class_objects = (
            ~true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been
        # 'detected'. So far, none

        true_class_boxes_detected = torch.zeros(
            (true_class_difficulties.size(0)),
            dtype=torch.uint8).to(device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(
            0)  #number of objects detected for this class
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(
            det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros(
            (n_class_detections),
            dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros(
            (n_class_detections),
            dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            # take the corresponding box and image for the detected object
            # we are considering
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their
            # difficulties, and whether they have been detected before.
            # in particular given a detected obj in this class c, consider
            # the other obj of the same class that you find in this_image to
            # which it belong.
            object_boxes = true_class_boxes[
                true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[
                true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false
            # positive.
            # It is necessary to have at least one obj, i.e. the obj in exam.
            # Otherwise is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image
            # of this class
            overlaps = find_jaccard_overlap(
                this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0),
                                         dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors
            # 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc.,
            # 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(
                true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5,
            # it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected,
                    # it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1
                        # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object
                    # is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than
            # the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection
        # in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives,
                                            dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives,
                                             dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
            cumul_true_positives + cumul_false_positives + 1e-10
        )  # cumul_precision--> size: (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects
        # cumul_recall--> size: (n_class_detections)
        # recall = TP/(TP+FN) --> at denominator we will have all
        # the objects that I want to detect (objs that are really on
        # the images, not difficult to detect)

        # Find the mean of the maximum of the precisions corresponding
        # to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1,
                                         step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)),
                                 dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()
        # c is in [1, n_classes - 1]
    return average_precisions


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels,
                  true_difficulties, label_map):
    '''
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    for an explanation

    :param list det_boxes: list of tensors, one tensor for each image
        containing detected objects' bounding boxes, thus the length of
        det_boxes corresponds to the number of images we are considering
        and the size of each tensor is (n_obj, 4), where n_obj is the
        number of objects detected for that image
    :param list det_labels: list of tensors, one tensor for each image
        containing detected objects' labels, thus the length of det_labels
        corresponds to the number of images we are considering and the
        size of each tensor is (n_obj), where n_obj is the number of
        objects detected for that image.
    :param list det_scores: list of tensors, one tensor for each image
        containing detected objects' labels' scores, thus the length of
        det_scores corresponds to the number of images we are considering
        and the size of each tensor is (n_obj), where n_obj is the number
        of objects detected for that image.
    :param list true_boxes: list of tensors, one tensor for each image
        containing actual objects' bounding boxes, thus the length of
        true_boxes corresponds to the number of images we are considering
        and the size of each tensor is (n_obj, 4), where n_obj is the
        number of objects for that image.
    :param list true_labels: list of tensors, one tensor for each image
        containing actual objects' labels, thus the length of true_labels
        corresponds to the number of images we are considering and the
        size of each tensor is (n_obj), where n_obj is the number of
        objects for that image.
    :param list true_difficulties: list of ByteTensors, one tensor for each
        image containing actual objects' difficulty (0 or 1), thus the
        length of true_difficulties corresponds to the number of images
        we are considering and the size of each tensor is (n_obj),
        where n_obj is the number of objects for that image.
    :param dict label_map: dictionary for the label map, where the
        keys are the labels of the objects(the classes) and their
        values the number of the classes to which they belong
        (0 for the background). Thus the length of this dict will be
        the number of the classes of the dataset.
    :return: average precisions for all classes, mean average
        precision (mAP)
    :rtype: list
    NOTE: n_obj detected may be different from n_obj true. For example,
          some objects could not have been detected. Or there can be
          multiple proposals for them (even if NMS has been used)
    '''
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(
        true_boxes) == len(true_labels) == len(true_difficulties)
    # these are all lists of tensors of the same length, i.e. number
    # of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping
    # track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device
    )  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping
    # track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(
        0) == det_scores.size(0)

    ####################################################
    # Calculate APs for each class (except background) #
    ####################################################
    average_precisions = calculate_AP(true_images, true_boxes,
                                      true_difficulties, true_labels,
                                      det_images, det_boxes, det_scores,
                                      det_labels, n_classes)

    ##########################################
    # Calculate Mean Average Precision (mAP) #
    ##########################################
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
    average_precisions = {
        rev_label_map[c + 1]: v
        for c, v in enumerate(average_precisions.tolist())
    }

    return average_precisions, mean_average_precision


#https://towardsdatascience.com/implementation-of-mean-average-precision-map-with-non-maximum-suppression-f9311eb92522
