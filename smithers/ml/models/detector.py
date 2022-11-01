'''
Module focused on the creation of the object detector and implementaion of the
training and testing phases.
'''
from functools import reduce
from pprint import PrettyPrinter
import time
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import ImageDraw, ImageFont

from smithers.ml.models.multibox_loss import MultiBoxLoss
from smithers.ml.models.utils import AverageMeter, clip_gradient, adjust_learning_rate, detect_objects, calculate_mAP, save_checkpoint_objdet 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Detector(nn.Module):
    '''
    Class that handles the creation of the Object Detector and its training and
    testing phases.
    '''
    def __init__(self, network, checkpoint, priors_cxcy, n_classes, epochs,
                 batch_size, print_freq, lr, decay_lr_at, decay_lr_to, momentum,
                 weight_decay, grad_clip, train_loader, test_loader):
        '''
    	:param list network: list of the different parts that compose the network
            For each element you need to construct it using the class related.
        :param path_file checkpoint: If None, you will need to initialize the
            model and optimizer from zero, otherwise you will load them from
            the checkpoint file given in input.
        :param tensor priors_cxcy: priors (default bounding boxes) in
            center-size coordinates, a tensor of size (n_boxes, 4)
        :param scalar n_classes: number of different type of objects in your
            dataset
        :param scalar epochs: number of epochs to run without early-stopping
        :param scalar batch_size: batch size
        :param int print_freq:  print training status every __ batches
        :param scalar lr: learning rate
        :param list decay_lr_at: decay learning rate after these many iterations
        :param float decay_lr_to: decay learnign rate to this fraction of the
            existing learning rate
        :param scalar momentum: momentum rate
        :param scalar weight_decay: weight decay
        :param bool grad_clip: clip if gradients are exploding, which may happen
            at larger batch sizes (sometimes at 32) - you will recognize it by a
            sorting error in the MultiBox loss calculation
        :param iterable train_loader: iterable object, it loads the dataset for
            training. It iterates over the given dataset, obtained combining a
            dataset(images, boxes and labels) and a sampler.
        :param iterable test_loader: iterable object, it loads the dataset for
            testing. It iterates over the given dataset, obtained combining a
            dataset(images, boxes and labels) and a sampler.
        :param bool reduced_network: if True it loads the forward function
            relative to the reduced detector, otherwise it loads the forward
            function of the full detector.
        '''
        super(Detector, self).__init__()

        self.priors = priors_cxcy.to(device)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.lr = lr
        self.decay_lr_at = decay_lr_at
        self.decay_lr_to = decay_lr_to
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.criterion = MultiBoxLoss(self.priors).to(device)
        #self.optimizer = self.init_optimizer()
        #Stocastic gradient descent
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.start_epoch, self.model, self.optimizer = self.load_network(
            network, checkpoint)
        # Since lower level features (conv4_3_feats) have considerably larger
        # scales, we take the L2 norm and rescale. Rescale factor is initially
        # set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(
            1, 512, 1, 1)).to(device)  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)
        self.epochs = self.start_epoch + epochs
        self.classifier = 'ssd'

    def load_network(self, network, checkpoint):
        '''
        Initialize model or load checkpoint
        If checkpoint is None, initialize the model and optimizer
        otherwise load checkpoint, coming from a previous training
	and load the model and optimizer from here
	:param list network: if is not None, it corresponds to a list
        containing the different structures that compose your net.
	    Otherwise, if None, it means that we are loading the model from
	    a checkpoint
	:param path_file checkpoint: If None, initialize the model and optimizer,
	    otherwise load them from the checkpoint file given in input.
        '''

        if checkpoint is None:
            start_epoch = 0
            model = [network[i].to(device) for i in range(len(network))]
            optimizer = self.init_optimizer(model, 'Adam')
        else:
            checkpoint = torch.load(checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
            net = checkpoint['model']
            model = [net[i].to(device) for i in range(len(net))]
            #model = [net[i].to(device) for i in range(len(net))]
            optimizer = checkpoint['optimizer']

        #Move to default device


#        model = model.to(device)
        return start_epoch, model, optimizer

    def init_optimizer(self, model, optim_str):
        '''
        Initialize the optimizer, with twice the default learning rate for
        biases, as in the original Caffe repo
        :param list model: list of the different parts that compose the network
            For each element you need to construct it using the class related.
        :param str optim_str: string defining the optimizer to use, e.g. 'SGD',
            'Adam'.
        :return optimizer: optimizer object chosen
        '''
        biases = list()
        not_biases = list()
        model_params = [model[i].named_parameters() for i in range(len(model))
                        ]
        for i in range(len(model_params)):
            for param_name, param in model_params[i]:
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        not_biases.append(param)
        if optim_str=='Adam':
            optimizer = torch.optim.Adam(params=[{
                'params': biases,
                'lr': self.lr
                }, {
                    'params': not_biases
                }],
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)
        elif optim_str=='SGD':
            optimizer = torch.optim.SGD(params=[{
                'params': biases,
                'lr': 2 * self.lr
                }, {
                    'params': not_biases
                }],
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        else:
            raise RuntimeError(
                'Invalid choice for the optimizer.')

        return optimizer

    def forward(self, images):
        '''
        Forward propagation of the entire network
        :param tensor images: dataset of images used
        :return: predicted localizations and classes scores (tensors) for
            each image
        '''
        images = images.to(device)   #dtype = torch.Tensor
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3, conv7 = self.model[0](images)
#        out_vgg = self.model[0](images)

        # Rescale conv4_3 (N, 512, 38, 38) after L2 norm
#        norm = conv4_3.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
#        conv4_3 = conv4_3 / norm  # (N, 512, 38, 38)
#        conv4_3 = conv4_3 * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)
        output_basenet = [conv4_3.to(device), conv7.to(device)]
#        output_basenet = [out_vgg]

        # Run auxiliary convolutions (higher level feature map generators)
        output_auxconv = self.model[1](conv7)
#        output_auxconv = self.model[1](out_vgg)
#        output_auxconv = self.model[1](out_vgg.view(out_vgg.size(0), -1)) 
#        output_auxconv = torch.unsqueeze(torch.unsqueeze(output_auxconv, dim=-1), dim=-1)
##        dim_kernel = int(np.sqrt(output_auxconv.size(1)))
##        output_auxconv = output_auxconv.view(output_auxconv.size(0), dim_kernel, dim_kernel)
##        output_auxconv = torch.unsqueeze(output_auxconv, dim=1)
###        output_auxconv = [output_auxconv.to(device)]
        # Run prediction convolutions (predict offsets w.r.t prior-boxes and
        # classes in each resulting localization box)
        locs, classes_scores = self.model[2](output_basenet, output_auxconv)

        return locs.to(device), classes_scores.to(device)


    def train_epoch(self, epoch):
        """
        One epoch's training.
        :param train_loader: an iterable over the given dataset, obtained
		    combining a dataset(images, boxes and labels) and a sampler.
        :param epoch: epoch number
        """
        for i in range(len(self.model) - 1):
            self.model[i].train()
#            self.model[i].features.train()
        self.model[-1].features_loc.train()
        self.model[-1].features_cl.train()
        #training mode enables dropout

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss

        start = time.time()

        # Batches
        for i, (images, boxes, labels, _) in enumerate(self.train_loader):
            data_time.update(time.time() - start)

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = self.forward(images)
            # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = self.criterion(predicted_locs, predicted_scores, boxes,
                                  labels)  # scalar

            # Backward prop.
            self.optimizer.zero_grad()
            #model.cleargrads()
            loss.backward()

            # Clip gradients, if necessary
            if self.grad_clip is not None:
                clip_gradient(self.optimizer, self.grad_clip)

            # Update model
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          epoch,
                          i,
                          len(self.train_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses))
        del predicted_locs, predicted_scores, images, boxes, labels
        # free some memory since their histories may be stored
        return loss.item()

    def train_detector(self, label_map = None): ##MODIF
        '''
        Total training of the detector for all the epochs
        '''
        print('Training has started.')
        # Epochs
        loss_values = []
        mAP_values = [] ##MODIF
        for epoch in range(self.start_epoch, self.epochs):

            # Decay learning rate at particular epochs
            if epoch in self.decay_lr_at:
                adjust_learning_rate(self.optimizer, self.decay_lr_to)

            # One epoch's training
            loss_val = self.train_epoch(epoch=epoch)
            loss_values.extend([loss_val])
            mAP_values.extend([10 * self.eval_detector(label_map, 'checkpoint_ssd300.pth.tar')]) ##MODIF
            if epoch%500 == 0: ##MODIF
                place_holder = save_checkpoint_objdet(epoch, self.model, self.optimizer, with_epochs = 'Yes') ##MODIF

        # Save checkpoint
        print('Training is now complete.')
        checkpoint_new = save_checkpoint_objdet(epoch, self.model, self.optimizer)
        return checkpoint_new, loss_values, #mAP_values  ##MODIF

    def train_detector_with_eval(self, label_map):
        '''
        Total training of the detector for all the epochs
        '''
        print('Training (with evaluation) has started.')
        # Epochs
        loss_values = []
        mAP_values = [] 
        for epoch in range(self.start_epoch, self.epochs):

            # Decay learning rate at particular epochs
            if epoch in self.decay_lr_at:
                adjust_learning_rate(self.optimizer, self.decay_lr_to)

            # One epoch's training
            loss_val = self.train_epoch(epoch=epoch)
            loss_values.extend([loss_val])
            mAP_values.extend([10 * self.eval_current_detector(label_map)]) ##MODIF
            if epoch%500 == 0: ##MODIF
                place_holder = save_checkpoint_objdet(epoch, self.model, self.optimizer, with_epochs = 'Yes') ##MODIF

        # Save checkpoint
        print('Training (with evaluation) is now complete.')
        checkpoint_new = save_checkpoint_objdet(epoch, self.model, self.optimizer)
        return checkpoint_new, loss_values, mAP_values  ##MODIF

    def eval_detector(self, label_map, checkpoint):
        '''
	Evaluation/Testing Phase

        :param dict label_map: dictionary for the label map, where the keys are
            the labels of the objects(the classes) and their values the number
            of the classes to which they belong (0 for the background). Thus the
            length of this dict will be the number of the classes of the
            dataset.
        :param str checkpoint: path to the checkpoint of the model obtained
            after the training phase
        '''
        # Load model checkpoint that is to be evaluated
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
#        model = [model[i].to(device) for i in range(len(model))]
#        model = checkpoint.model
        # set the network(all classes derived from nn.module) in evaluation mode
        for i in range(len(model) - 1):
            model[i].eval()
            #model[i].features.eval()
        model[-1].features_loc.eval()
        model[-1].features_cl.eval()
        # Lists to store detected and true boxes, labels, scores
        det_boxes = list()
        det_labels = list()
        det_scores = list()
        true_boxes = list()
        true_labels = list()
        true_difficulties = list()
        # it is necessary to know which objects are 'difficult', see
        # 'calculate_mAP' in utils.py

        # Good formatting when printing the APs for each class and mAP
        pp = PrettyPrinter()

        #torch.no_grad() impacts the autograd engine and deactivate it.
        #It will reduce memory usage and speed up computations but you
        #would not be able to backprop (which you do not want in an eval
        #script).
        with torch.no_grad():
            # Batches
            for i, (images, boxes, labels, difficulties) in enumerate(
                    tqdm(self.test_loader, desc='Evaluating')):
                images = images.to(device)  # (N, 3, 300, 300)

                # Forward prop.
                predicted_locs, predicted_scores = self.forward(images)

                # Detect objects in SSD output
#                print('priors:', self.priors.size())
#                print('predicted_locs', predicted_locs.size())
#                print('predicted_scores', predicted_scores.size())
                det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(
                    self.priors,
                    predicted_locs,
                    predicted_scores,
                    self.n_classes,
                    min_score=0.01,
                    max_overlap=0.45,
                    top_k=20)
                # Evaluation MUST be at min_score=0.01, max_overlap=0.45,
                # top_k=200 for fair comparision with the paper's results
                # and other repos

                # Store this batch's results for mAP calculation
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]
                difficulties = [d.to(device) for d in difficulties]

                det_boxes.extend(det_boxes_batch)
                det_labels.extend(det_labels_batch)
                det_scores.extend(det_scores_batch)
                true_boxes.extend(boxes)
                true_labels.extend(labels)
                true_difficulties.extend(difficulties)

            # Calculate mAP
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores,
                                     true_boxes, true_labels, true_difficulties,
                                     label_map)
        print(APs)
        # Print AP for each class
        pp.pprint(APs)

        print('\nMean Average Precision (mAP): %.3f' % mAP)
        return mAP ##MODIF

    def eval_current_detector(self, label_map):
        '''
	Evaluation/Testing Phase

        :param dict label_map: dictionary for the label map, where the keys are
            the labels of the objects(the classes) and their values the number
            of the classes to which they belong (0 for the background). Thus the
            length of this dict will be the number of the classes of the
            dataset.
        :param str checkpoint: path to the checkpoint of the model obtained
            after the training phase
        '''
        # Load model checkpoint that is to be evaluated
        model = self.model
#        model = [model[i].to(device) for i in range(len(model))]
#        model = checkpoint.model
        # set the network(all classes derived from nn.module) in evaluation mode
        for i in range(len(model) - 1):
            model[i].eval()
            #model[i].features.eval()
        model[-1].features_loc.eval()
        model[-1].features_cl.eval()
        # Lists to store detected and true boxes, labels, scores
        det_boxes = list()
        det_labels = list()
        det_scores = list()
        true_boxes = list()
        true_labels = list()
        true_difficulties = list()
        # it is necessary to know which objects are 'difficult', see
        # 'calculate_mAP' in utils.py

        # Good formatting when printing the APs for each class and mAP
        pp = PrettyPrinter()

        #torch.no_grad() impacts the autograd engine and deactivate it.
        #It will reduce memory usage and speed up computations but you
        #would not be able to backprop (which you do not want in an eval
        #script).
        with torch.no_grad():
            # Batches
            for i, (images, boxes, labels, difficulties) in enumerate(self.test_loader):
                images = images.to(device)  # (N, 3, 300, 300)

                # Forward prop.
                predicted_locs, predicted_scores = self.forward(images)

                # Detect objects in SSD output
#                print('priors:', self.priors.size())
#                print('predicted_locs', predicted_locs.size())
#                print('predicted_scores', predicted_scores.size())
                det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(
                    self.priors,
                    predicted_locs,
                    predicted_scores,
                    self.n_classes,
                    min_score=0.01,
                    max_overlap=0.45,
                    top_k=20)
                # Evaluation MUST be at min_score=0.01, max_overlap=0.45,
                # top_k=200 for fair comparision with the paper's results
                # and other repos

                # Store this batch's results for mAP calculation
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]
                difficulties = [d.to(device) for d in difficulties]

                det_boxes.extend(det_boxes_batch)
                det_labels.extend(det_labels_batch)
                det_scores.extend(det_scores_batch)
                true_boxes.extend(boxes)
                true_labels.extend(labels)
                true_difficulties.extend(difficulties)

            # Calculate mAP
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores,
                                     true_boxes, true_labels, true_difficulties,
                                     label_map)
        #print(APs)
        # Print AP for each class
        #pp.pprint(APs)

        print('\nMean Average Precision (mAP): %.3f' % mAP)
        return mAP ##MODIF

    def detect(self,
               original_image,
               checkpoint,
               label_map,
               min_score,
               max_overlap,
               top_k,
               suppress=None):
        """
        Detect objects in an image with a trained SSD300, and visualize
        the results.

        :param PIL Imagw original_image: image, a PIL Image
        :param str checkpoint: path to the checkpoint of the model obtained
            after the training phase
        :param dict label_map: dictionary for the label map, where the keys are
            the labels of the objects(the classes) and their values the number
            of the classes to which they belong (0 for the background). Thus the
            length of this dict will be the number of the classes of the
            dataset.
        :param float min_score: minimum threshold for a detected box to
            be considered a match for a certain class
        :param float max_overlap: maximum overlap two boxes can have so
            that the one with the lower score is not suppressed via
            Non-Maximum Suppression (NMS)
        :param int top_k: if there are a lot of resulting detection across
            all classes, keep only the top 'k'
        :param list suppress:a list of classes that you know for sure cannot be
            in the image or you do not want in the image. If None, it does not
             suppress anything.
        :return: annotated image, a PIL Image
        """

        # Load model checkpoint that is to be evaluated
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        model = [model[i].to(device) for i in range(len(model))]
        # set the network(all classes derived from nn.module) in evaluation mode
        for i in range(len(model) - 1):
            model[i].eval()
#            model[i].features.eval()
        model[-1].features_loc.eval()
        model[-1].features_cl.eval()

        # Color map for bounding boxes of detected objects from
        # https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
        distinct_colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4',
            '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', '#008080', '#000080',
            '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
            '#e6beff', '#808080', '#FFFFFF'
        ]
        label_color_map = {
            k: distinct_colors[i]
            for i, k in enumerate(label_map.keys())
        }

        # Transforms
        resize = transforms.Resize((300, 300))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(to_tensor(resize(original_image)))

        # Move to default device
        image = image.to(device)

        # Forward prop.
        predicted_locs, predicted_scores = self.forward(image.unsqueeze(0))

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = detect_objects(
            self.priors,
            predicted_locs,
            predicted_scores,
            self.n_classes,
            min_score=min_score,
            max_overlap=max_overlap,
            top_k=top_k)

        # Move detections to the CPU
        det_boxes = det_boxes[0].to(device)

        # Transform to original image dimensions
        original_dims = torch.FloatTensor([
            original_image.width, original_image.height, original_image.width,
            original_image.height
        ]).unsqueeze(0).to(device)
        det_boxes = det_boxes * original_dims

        rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
        # Decode class integer labels
        det_labels = [
            rev_label_map[l] for l in det_labels[0].to(device).tolist()
        ]

        # If no objects found, the detected labels will be set to ['0.'], i.e.
        # ['background'] in detect_objects() in util.py
        if det_labels == ['background']:
            # Just return original image
            return original_image

        # Annotate
        annotated_image = original_image
        # Create an object that can be used to draw in the given image.
        draw = ImageDraw.Draw(annotated_image)
        #font = ImageFont.truetype("./calibril.ttf", 15)
        # this line does not work for me, try to fix in case, use dafault font
        font = ImageFont.load_default()

        # Suppress specific classes, if needed
        for i in range(det_boxes.size(0)):
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue

            # Boxes
            box_location = det_boxes[i].tolist()
            draw.rectangle(xy=box_location,
                           outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location],
                           outline=label_color_map[det_labels[i]])
            # a second rectangle at an offset of 1 pixel to increase line
            # thickness
            # draw.rectangle(xy=[l + 2. for l in box_location],
            #                outline=label_color_map[det_labels[i]])
            # a third rectangle at an offset of 1 pixel to increase line
            # thickness
            # draw.rectangle(xy=[l + 3. for l in box_location],
            #                outline=label_color_map[det_labels[i]])
            # a fourth rectangle at an offset of 1 pixel to increase line
            # thickness

            # Text
            text_size = font.getsize(det_labels[i].upper())
            text_location = [
                box_location[0] + 2., box_location[1] - text_size[1]
            ]
            textbox_location = [
                box_location[0], box_location[1] - text_size[1],
                box_location[0] + text_size[0] + 4., box_location[1]
            ]
            draw.rectangle(xy=textbox_location,
                           fill=label_color_map[det_labels[i]])
            draw.text(xy=text_location,
                      text=det_labels[i].upper(),
                      fill='white',
                      font=font)
        img_final = annotated_image.save('out.jpg')
        del draw

        return annotated_image


class Reduced_Detector(Detector):
    '''
    Class that handles the creation of the Reduced Object Detector and its training and
    testing phases. This class extends the Detector class.
    '''
    '''def __init__(self, network, checkpoint, priors_cxcy, n_classes, epochs,
                 batch_size, print_freq, lr, decay_lr_at, decay_lr_to, momentum,
                 weight_decay, grad_clip, train_loader, test_loader):
        super(Reduced_Detector, self).__init__(self, network, checkpoint, priors_cxcy, n_classes, epochs,
                 batch_size, print_freq, lr, decay_lr_at, decay_lr_to, momentum,
                 weight_decay, grad_clip, train_loader, test_loader)'''

    def forward(self, images):
        '''
        Forward propagation of the entire network
        :param tensor images: dataset of images used
        :return: predicted localizations and classes scores (tensors) for
            each image
        '''
        images = images.to(device)   #dtype = torch.Tensor
        # Run VGG base network convolutions (lower level feature map generators)

#       conv4_3, conv7 = self.model[0](images)
        out_vgg = self.model[0](images)
        #CAMBIA COME DATO OUTPUT IN VGG, PIU' COMODA LISTA--. CAMBIARE IN TUTTI
        #I FILE!

        # Rescale conv4_3 (N, 512, 38, 38) after L2 norm
#        norm = conv4_3.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
#        conv4_3 = conv4_3 / norm  # (N, 512, 38, 38)
#        conv4_3 = conv4_3 * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)
#        output_basenet = [conv4_3.to(device), conv7.to(device)]
        output_basenet = [out_vgg]

        # Run auxiliary convolutions (higher level feature map generators)
#        output_auxconv = self.model[1](conv7)
#        output_auxconv = self.model[1](out_vgg)
        output_auxconv = self.model[1](out_vgg)
#        output_auxconv = self.model[1](out_vgg.view(out_vgg.size(0), -1)) #no hosvd
        
#        output_auxconv = torch.unsqueeze(torch.unsqueeze(output_auxconv, dim=-1), dim=-1) #no hosvd
##        dim_kernel = int(np.sqrt(output_auxconv.size(1)))
##        output_auxconv = output_auxconv.view(output_auxconv.size(0), dim_kernel, dim_kernel)
##        output_auxconv = torch.unsqueeze(output_auxconv, dim=1)
        output_auxconv = [output_auxconv.to(device)]
        # Run prediction convolutions (predict offsets w.r.t prior-boxes and
        # classes in each resulting localization box)
        locs, classes_scores = self.model[2](output_basenet, output_auxconv)

        return locs.to(device), classes_scores.to(device)