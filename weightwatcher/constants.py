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

OVER_TRAINED_THRESH = 6.0
UNDER_TRAINED_THRESH = 2.0
            

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

DEFAULT_START_ID = 0
    
# These are NOT the defaults...see analyze() for actual defaults
DEFAULT_PARAMS = {GLOROT_FIX: False, NORMALIZE:False, CONV2D_NORM:True, RANDOMIZE: True, 
                  SAVEDIR:DEF_SAVE_DIR, SAVEFIG:True, RESCALE:True, PLOT:False,
                  DELTA_ES:False, INTRA:False, CHANNELS_STR:None, FFT:False,  CONV2D_FFT:False, 
                  WW2X:DEFAULT_WW2X, POOL:DEFAULT_POOL, VECTORS:True, SMOOTH:None, STACKED:False, 
                  SVD_METHOD:ACCURATE_SVD,  
                  FIX_FINGERS:None, FIT:POWER_LAW, MAX_FINGERS:DEFAULT_MAX_FINGERS, FINGER_THRESH:DEFAULT_FINGER_THRESH,
                  SPARSIFY: True, DETX: True,  MP_FIT:False,
                  MIN_EVALS:DEFAULT_MIN_EVALS, MAX_EVALS:DEFAULT_MAX_EVALS, MAX_N:DEFAULT_MAX_N,
                  TOLERANCE:WEAK_RANK_LOSS_TOLERANCE, START_IDS:DEFAULT_START_ID, ADD_BIASES:False,
                  PL_PACKAGE:DEFAULT_POWERLAW_PACKAGE, XMAX:DEFAULT_XMAX}


EVALS_THRESH = 0.00001
ALMOST_ZERO = 1e-10

class LAYER_TYPE(IntFlag):
    UNKNOWN = auto()
    STACKED = auto()
    DENSE = auto()
    CONV1D = auto()
    CONV2D = auto()
    FLATTENED = auto()
    EMBEDDING = auto()
    NORM = auto()

# framework names
KERAS = 'kers'
PYTORCH = 'pytorch'
PYSTATEDICT = 'pystatedict'
ONNX = 'onnx'

class FRAMEWORK(IntFlag):
    UNKNOWN = auto()
    PYTORCH = auto()
    KERAS = auto()
    ONNX = auto()
    PYSTATEDICT = auto()
    PYSTATEDICTFILE = auto()
    KERASH5 = auto()
    KERASH5FILE = auto()


class CHANNELS(IntFlag):
    UNKNOWN = auto()
    FIRST = auto()
    LAST = auto()   
            
class METHODS(IntFlag):
    DESCRIBE = auto()
    ANALYZE = auto()
            
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
