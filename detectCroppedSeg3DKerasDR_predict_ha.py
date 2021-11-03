import sys
import os
from os import path
#sys.path.insert(0, '/fileserver/abd/github_ha_editing/')

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ##P4000=='0', P6000=='1'

import funcs_ha_use
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import scipy
from scipy.ndimage import zoom
from scipy import signal
from scipy.interpolate import interp1d
import pickle
from sklearn.metrics import precision_recall_curve
from skimage import morphology
from keras.models import Model,load_model,Sequential
from networks_ah import get_unet2, get_rbunet, get_meshNet, get_denseNet, calculatedPerfMeasures
from networks_ah import get_unetCnnRnn
from networks_ah import get_denseNet103, get_unet3
#from selectTrainAndTestSubjects_ha_2_use import selectTrainAndTestSubjects

reconMethod = 'SCAN';

def singlePatientDetection(pName, baseline, params, organTarget):
    write('Step 1')
   
    return 0

