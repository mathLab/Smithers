'''
Module focused on the implementation of the MultiBox Loss Function.
'''
import torch
import torch.nn as nn
import numpy as np

from smithers.ml.models.utils import cxcy_to_xy, find_jaccard_overlap, cxcy_to_gcxgcy, xy_to_cxcy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
	As described in the SSD original paper:
	'SSD: Single Shot Multibox Detector' by
    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
    Scott Reed, Cheng-Yang Fu, Alexander C. Berg
    https://arxiv.org/abs/1512.02325
    DOI:	10.1007/978-3-319-46448-0_2

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """
    def __init__(
            self,
            priors_cxcy,
            threshold=0.5,
            neg_pos_ratio=3,
            alpha=1.,
            #size_average=None,
            #reduce=None,
            #reduction='mean'):
        reduction='None'):
        '''
	:param tensor priors_cxcy: priors (default bounding boxes) in center-size
            coordinates, a tensor of size (n_boxes, 4)
	:param float threshold: Threshold for the Jaccard overlap. If it is greater
            than the threshold value, the box is said to "contain" the object.
            Thu we have a positive match. Otherwise, it does not contain it and
            we have a negative match and it is labeled as background.
	:param int neg_pos_ratio: ratio that connected the number of positive matches
            (N_p) with the hard negatives (N_hn). Usually they are a fixed
            multiple of the number of positive matches for an image:
            N_hn = neg_pos_ratio * N_p
	:param int alpha: ratio of combination between the two different losses
            (localization and confidence) It can be a learnable parameter or
            fixed, as in the case of SSD300 where the authors decided to use
            alpha=1.(default value)
        :param string reduction: Specifies the reduction to apply to the output:
            'None', 'mean' or 'sum'.
	     - If None, no reduction will be applied.
	     - If mean, the sum of the output will be divided by the number of
               elements in the output.
             - If sum, the output will be summed.
	'''
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        #convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to
        #boundary coordinates (x_min, y_min, x_max, y_max)
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        #self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior
            boxes, a tensor of dimensions (N, 8732, 4). Thus, one of the
            outputs of PredictionConvolutions.forward().
        :param predicted_scores: class scores for each of the encoded
            locations/boxes, a tensor of dimensions (N, 8732, n_classes). Thus,
            the other output of PredictionConvolutions.forward().
        :param boxes: true object bounding boxes (ground-truth) in boundary
            coordinates, a list of N tensors, where N is the total number of
            pictures. (for each image I have a n_objects boxes, where
            n_objects is the number of objects contained in that image)
        :param labels: true object labels, a list of N tensors, where each
            tensor has dimensions n_objects(for that image).
        :return: multibox loss, a zero dimensional tensor (NOT a scalar!)
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        print(f'n_priors è {n_priors}\npredicted_locs.size(1) è {predicted_locs.size(1)}\npredicted_scores.size(1) è {predicted_scores.size(1)}')
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4),
                                dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors),
                                   dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(
                dim=0)  # (8732)

            # We don't want a situation where an object is not represented in
            # our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is
            #    therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based
            #    on the threshold (0.5).
            # To remedy: First, find the prior that has the maximum overlap for
            # each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding
            # maximum-overlap-prior.
            # This fixes 1. : in this way all the objects are considered.
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(
                range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap
            # of greater than 0.5. This fixes 2.: the objects that previously
            # where not considered may have an overlap lower than the
            # threshold. Thus in order to avoid this, we give them a
            # value of 1.
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than
            # the threshold to be background (no object)
            label_for_each_prior[
                overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store classes and localizations
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed
            # predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(
                xy_to_cxcy(boxes[i][object_for_each_prior]),
                self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732) boolean values

        # LOCALIZATION LOSS
        # Localization loss is computed only over positive (non-background)
        # priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors],
                                  true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor
        # when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4),
        # predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS
        # Confidence loss is computed over positive priors and the most
        # difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE, we will take the hardest
        # (neg_pos_ratio * n_positives) negative priors, i.e where there is
        # maximum loss. This is called Hard Negative Mining - it concentrates on
        # hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        # we flatten the tensor predicted_scores from
        # (num_img, num_priors, n_classes) to (num_img * num_priors, n_classes)
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes),
                                           true_classes.view(-1))  # (N * 8732)

        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of
        # decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.
        # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(
            dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        # expand_as: used to give the same dimension of a tensor1 to a tensor2
        # which shape differs from those of tensor1 in a dimension (the value
        # associated for this dimension in tensor2 has to be less than its value
        # for tensor1)
        hardness_ranks = torch.LongTensor(
            range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(
                device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(
            1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[
            hard_negatives]  # (sum(n_hard_negatives)) 1d tensor
        # As in the paper, averaged over positive priors only, although computed
        # over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()
                     ) / n_positives.sum().float()  # (), scalar
        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss
