'''
Object_Detector init
'''
__project__ = 'Object_Detector'
__title__ = 'object_detector'
__author__ = 'Laura Meneghetti, Nicola Demo'
__maintainer__ = __author__

from smithers.ml.models.vgg import VGG
from smithers.ml.models.aux_conv import AuxiliaryConvolutions
from smithers.ml.models.predictor import PredictionConvolutions
from smithers.ml.models.multibox_loss import MultiBoxLoss
from smithers.ml.models.detector import Detector
from smithers.ml.models.utils_imagerec import *
from smithers.ml.models.hosvd import HOSVD
from smithers.ml.models.ahosvd import AHOSVD
