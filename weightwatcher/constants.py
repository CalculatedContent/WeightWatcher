# Copyright 2018 Calculation Consulting [calculationconsulting.com]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
=====================================================================================================================
This file defines constants, enums, and global variables used by the WeightWatcher module. Let's break down its content:
1. Constants and Global Variables
WW_NAME: The name of the WeightWatcher module.
ERROR: Represents an error code, usually -1.
DEF_SAVE_DIR: The default directory for saving figures, "ww-img".
LAYERS: Key used in configuration dictionaries for specifying layer filters.
START_IDS: Key for setting the starting layer ID counter (0 or 1).
TRUNCATED_SVD, ACCURATE_SVD, FAST_SVD, SVD_METHOD: Constants related to singular value decomposition (SVD) methods.
VALID_SVD_METHODS: List of valid SVD methods.
XMIN_PEAK, XMIN_MAX, CLIP_XMAX, MAX_FINGERS, DEFAULT_MAX_FINGERS: Constants for handling "finger" issues in power law fits.
SVD, RMT: Constants for SVD-based smoothing methods.
FIT, PL, POWER_LAW, PL_PACKAGE, POWERLAW_PACKAGE, WW_POWERLAW_PACKAGE, WW_POWERLAW: Constants related to power law fitting and the package used (either powerlaw or ww).
WW_CONFIG_FILENAME: The name of the configuration file for PyTorch state dictionaries ("ww_config.json").
PYTORCH, TENSORFLOW: Framework names.
XMAX, XMAX_FORCE, FORCE, DEFAULT_XMAX: Constants related to the maximum eigenvalue range for power law fits.
TPL, TRUNCATED_POWER_LAW, FIX_FINGERS, E_TPL: Constants related to truncated power law fitting and finger fixing.
LOG_NORMAL, EXPONENTIAL: Other potential distributions for fitting.
EUCLIDEAN, CKA, RAW: Distance calculation methods.
SUCCESS, FAILED, OVER_TRAINED, UNDER_TRAINED, OVER_TRAINED_THRESH, UNDER_TRAINED_THRESH: Constants for representing analysis status and over/under-training indicators.
UNKNOWN: Represents an unknown value.
GLOROT_FIX, NORMALIZE: Flags for applying Glorot normalization or general normalization.
SAVEFIG, SAVEDIR: Keys for saving figures and specifying the save directory.
DELTA_ES, INTRA, FFT, CONV2D_NORM, CONV2D_FFT, FINGER_THRESH, DEFAULT_FINGER_THRESH: Flags and constants for various analysis and transformation options.
GLOROT_FIT: Flag for applying Glorot normalization.
WW2X, POOL: Deprecated constants; POOL is the replacement for WW2X.
PEFT: Flag for handling PEFT/LoRA models.
VECTORS, SMOOTH, MP_FIT: Flags for analyzing eigenvectors, smoothing, and performing MP fitting.
DEFAULT_WW2X, DEFAULT_POOL: Default values for WW2X and POOL.
RESCALE, RANDOMIZE, SPARSIFY, DETX, LAMBDA_MIN, AUTO: Constants related to rescaling, randomization, sparsification, and SVD smoothing methods.
MIN_EVALS, DEFAULT_MIN_EVALS, MIN_NUM_EVALS, MAX_EVALS, DEFAULT_MAX_EVALS, MAX_NUM_EVALS, MAX_N, DEFAULT_MAX_N: Constants for specifying minimum and maximum numbers of eigenvalues.
PLOT, STACKED: Flags for enabling plotting and stacked analysis.
CHANNELS_STR, FIRST, LAST: Keys and values for handling channel order.
TOLERANCE, WEAK_RANK_LOSS_TOLERANCE: Tolerance values for rank loss computations.
ADD_BIASES: Flag for adding biases.
DEFAULT_PEFT, PEFT_WITH_BASE: Default and specific values for handling PEFT/LoRA models.
INVERSE: Flag for computing the inverse ESD.
DEFAULT_START_ID: Default starting layer ID.
DEFAULT_PARAMS: Dictionary of default parameter settings.
EVALS_THRESH, EPSILON, EVALS_HALF_THRESH, HALF_EPSILON: Thresholds and epsilon values used in precision computations.
2. Enums
LAYER_TYPE: Enum for representing different layer types.
FRAMEWORK: Enum for representing deep learning frameworks.
CHANNELS: Enum for representing channel order.
METHODS: Enum for representing analysis methods (DESCRIBE or ANALYZE).
MODEL_FILE_FORMATS: Enum for representing different model file formats.
METRICS: Enum for representing different metrics calculated.
XMIN: Enum for representing different options for setting the minimum eigenvalue.
3. Layer Types
LAYER_TYPES_SUPPORTED: List of supported layer types.
4. Framework Names
KERAS, KERAS_H5_FILE, PYTORCH, PYSTATEDICT, ONNX, WW_FLATFILES, SAFETENSORS: String representations of framework names.
5. Channel Order
CHANNELS: Enum for handling channel order.
6. Analysis Methods
METHODS: Enum for representing analysis methods.
7. Model File Formats
MODEL_FILE_FORMATS: Enum for representing different model file formats.
8. Metrics
METRICS: Enum for representing calculated metrics.
9. Minimum Eigenvalue Options
XMIN: Enum for minimum eigenvalue settings.
In essence, this file provides a comprehensive set of constants, enums, and global variables that organize the information and functionality of the WeightWatcher module. These elements ensure consistency and readability within the module's code.

=====================================================================================================================================================================================================================================================
'''

from enum import IntFlag, auto, Enum

WW_NAME = 'weightwatcher'

ERROR = -1

DEF_SAVE_DIR = 'ww-img'

LAYERS = 'layers'
START_IDS = 'layer_ids_start' # 0 | 1

TRUNCATED_SVD = 'truncated'
TRUNCATED = 'truncated'
ACCURATE_SVD = 'accurate'
ACCURATE = 'accurate'
FAST_SVD = 'fast'
FAST = 'fast'
SVD_METHOD = 'svd_method'
VALID_SVD_METHODS = [FAST_SVD, ACCURATE_SVD]    # Add TRUNCATED_SVD here when ready to enable

# fi_ fingers options
XMIN_PEAK = 'xmin_peak'
XMIN_MAX = 'xmin_max'
CLIP_XMAX = 'clip_xmax'
MAX_FINGERS = 'MAX_FINGERS'
DEFAULT_MAX_FINGERS = 10


SVD = 'svd' # TruncatedSVD for Smoothing
RMT = 'rmt' # pyRMT / RIE Removed for 0.6.5

# Power Law Fitting methods
#  xmax-Npne is now default
FIT = 'fit'
PL = 'PL'
POWER_LAW = 'power_law'
POWERLAW = 'powerlaw'

PL_PACKAGE = 'pl_package'
POWERLAW_PACKAGE = 'powerlaw'
WW_POWERLAW_PACKAGE = 'ww'
WW_POWERLAW = WW_POWERLAW_PACKAGE

DEFAULT_POWERLAW_PACKAGE = WW_POWERLAW_PACKAGE


WW_CONFIG_FILENAME = "ww_config.json"
PYTORCH = 'pytorch'
TENSORFLOW = 'tensorflow'


# forcing xmax only available for older powerlaw package 

XMAX = 'xmax'
XMAX_FORCE = 'force' 
FORCE = 'force' 
DEFAULT_XMAX = None


TPL = 'TPL'
TRUNCATED_POWER_LAW = 'truncated_power_law'
FIX_FINGERS = 'fix_fingers'
E_TPL = 'E_TPL' #extended power law

#
LOG_NORMAL = 'lognormal'
EXPONENTIAL = 'exponential'

# distance choices
EUCLIDEAN = 'euclidean'
CKA = 'cka'
RAW = 'raw'

# STATUSes

SUCCESS = 'success'
FAILED = 'failed'
OVER_TRAINED = 'over-trained'
UNDER_TRAINED = 'under-trained'

OVER_TRAINED_THRESH = 2.0
UNDER_TRAINED_THRESH = 6.0
            

UNKNOWN = 'unknown'

GLOROT_FIX = 'glorot_fix'
NORMALIZE = 'normalize'

LAYERS = 'layers'

SAVEFIG = 'savefig'
SAVEDIR = 'savedir'

DELTA_ES = 'deltaEs'
INTRA = 'intra'
FFT = 'fft'
CONV2D_NORM = 'conv2d_norm'
CONV2D_FFT = 'conv2d_fft'

FINGER_THRESH = 'finger_thresh'
DEFAULT_FINGER_THRESH = 1.0

GLOROT_FIT = 'glorot_fit'

WW2X = 'ww2x'  # deprecated
POOL = 'pool'

PEFT = 'peft'


VECTORS = 'vectors'
SMOOTH = 'smooth'
MP_FIT = 'mp_fit'

# maybe should be True
DEFAULT_WW2X = False # deprecated
DEFAULT_POOL = True

RESCALE = 'rescale'
RANDOMIZE = 'randomize'
SPARSIFY = 'sparsify'
DETX = 'detX' # compute detx and for smoothing
LAMBDA_MIN = 'alpha_min' # smoothing
AUTO = 'auto' # smoothing


MIN_EVALS = 'min_evals'
DEFAULT_MIN_EVALS = 10
MIN_NUM_EVALS = 10

MAX_EVALS = 'max_evals'
DEFAULT_MAX_EVALS = 15000
MAX_NUM_EVALS= DEFAULT_MAX_EVALS

MAX_N = 'max_N'
DEFAULT_MAX_N = 50000

PLOT = 'plot'
STACKED = 'stacked'

CHANNELS_STR = 'channels'
FIRST = 'first'
LAST = 'last'   

TOLERANCE = 'tolerance'
WEAK_RANK_LOSS_TOLERANCE = 0.000001 # on ei=gen values


ADD_BIASES = 'add_biases'

DEFAULT_PEFT = False # | True | 'peft' | 'peft_onlu'
#PEFT_ONLY = 'peft_only'  #deprcated
PEFT_WITH_BASE = 'with_base'

INVERSE = 'inverse'



DEFAULT_START_ID = 0 # this is hard coded to 1 for PyStateDict
    
# These are NOT the defaults...see analyze() for actual defaults
DEFAULT_PARAMS = {GLOROT_FIX: False, NORMALIZE:False, CONV2D_NORM:True, RANDOMIZE: True, 
                  SAVEDIR:DEF_SAVE_DIR, SAVEFIG:True, RESCALE:True, PLOT:False,
                  DELTA_ES:False, INTRA:False, CHANNELS_STR:None, FFT:False,  CONV2D_FFT:False, 
                  WW2X:DEFAULT_WW2X, POOL:DEFAULT_POOL, VECTORS:True, SMOOTH:None, STACKED:False, 
                  SVD_METHOD:FAST_SVD,  
                  FIX_FINGERS:None, FIT:POWER_LAW, MAX_FINGERS:DEFAULT_MAX_FINGERS, FINGER_THRESH:DEFAULT_FINGER_THRESH,
                  SPARSIFY: True, DETX: True,  MP_FIT:False,
                  MIN_EVALS:DEFAULT_MIN_EVALS, MAX_EVALS:DEFAULT_MAX_EVALS, MAX_N:DEFAULT_MAX_N,
                  TOLERANCE:WEAK_RANK_LOSS_TOLERANCE, START_IDS:DEFAULT_START_ID, ADD_BIASES:False, XMIN_MAX:None,
                  PL_PACKAGE:DEFAULT_POWERLAW_PACKAGE, XMAX:DEFAULT_XMAX, PEFT:DEFAULT_PEFT, INVERSE:False}


EVALS_THRESH =1e-5
EPSILON = 1e-10 # only in normal precision

EVALS_HALF_THRESH = 1e-4
HALF_EPSILON = 6e-05 # torch half float precision

UNKNOWN = 'unknown'
NORM = 'norm'
DENSE = 'dense'
CONV2D = 'conv2d'
CONV1D = 'conv1d'
MBEDDING = 'embedding'

class LAYER_TYPE():
    UNKNOWN = UNKNOWN
    STACKED = 'stacked'
    DENSE = DENSE
    CONV1D = CONV1D
    CONV2D = CONV2D
    FLATTENED = 'flattened'
    EMBEDDING = 'embedding'
    NORM = NORM
    
LAYER_TYPES_SUPPORTED = [LAYER_TYPE.CONV2D, LAYER_TYPE.CONV1D, LAYER_TYPE.DENSE, LAYER_TYPE.EMBEDDING, LAYER_TYPE.NORM]


# framework names
KERAS = 'keras'
KERAS_H5_FILE = 'keras_h5_file'

PYTORCH = 'pytorch'
PYSTATEDICT = 'pystatedict'
ONNX = 'onnx'
WW_FLATFILES = "ww_flatfiles"
PYTORCH = "pytorch"
SAFETENSORS = "safetensors"



# class FRAMEWORK(IntFlag):
#     UNKNOWN = auto()
#     PYTORCH = auto()
#     KERAS = auto()
#     ONNX = auto()
#     PYSTATEDICT = auto()
#     PYSTATEDICT_DIR = auto()
#     WW_FLATFILES = auto()
#     KERASH5 = auto()
#     KERASH5FILE = auto()
    
class FRAMEWORK():
    UNKNOWN = UNKNOWN
    PYTORCH = 'pytorch'
    KERAS = 'keras'
    ONNX = 'onnx'
    PYSTATEDICT = 'pystatedict'
    PYSTATEDICT_DIR = 'pystatedict_dir'
    WW_FLATFILES = WW_FLATFILES
    KERAS_H5_FILE = 'keras_h5_file'


class CHANNELS():
    UNKNOWN = UNKNOWN
    FIRST = 'first'
    LAST = 'last' 
            
class METHODS(IntFlag):
    DESCRIBE = auto()
    ANALYZE = auto()
    
# only used to extract into ww_flatfiels format        
class MODEL_FILE_FORMATS():
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    WW_FLATFILES = WW_FLATFILES
    KERAS_H5_FILE = 'keras_h5_file'

            
# TODO either complete or remove thi 
class METRICS():
    NORM = "norm"
    LOG_NORM = "log_norm"
    ALPHA = "alpha"
    ALPHA_WEIGHTED = "alpha_weighted"
    LOG_ALPHA_NORM = "log_alpha_norm"
    SPECTRAL_NORM = "spectral_norm"
    LOG_SPECTRAL_NORM = "log_spectral_norm"
    STABLE_RANK = "stable_rank"
    MP_SOFTRANK = "mp_softrank"
    MATRIX_RANK = "matrix_rank"
    MATRIX_ENTROPY = "entropy"


class XMIN(IntFlag):
    UNKNOWN = auto()
    AUTO = auto()
