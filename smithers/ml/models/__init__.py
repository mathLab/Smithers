'''
Object_Detector init
'''
__project__ = 'Object_Detector'
__title__ = 'object_detector'
__author__ = 'Laura Meneghetti, Nicola Demo'
__maintainer__ = __author__

from models.vgg import VGG
from models.aux_conv import AuxiliaryConvolutions
from models.predictor import PredictionConvolutions
from models.multibox_loss import MultiBoxLoss
from models.detector import Detector
from models.utils import *
