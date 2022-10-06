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


DEF_SAVE_DIR = 'ww-img'

LAYERS = 'layers'
START_IDS = 'layer_ids_start' # 0 | 1

TRUNCATED_SVD = 'truncated_svd'
FULL_SVD = 'full_svd'

# fi_ fingers options
XMIN_PEAK = 'xmin_peak'
CLIP_XMAX = 'clip_xmax'

SVD = 'svd' # TruncatedSVD for Smoothing
RMT = 'rmt' # pyRMT / RIE Removed for 0.6.5

PL = 'PL'
TPL = 'TPL'
E_TPL = 'E_TPL' #extended power law
TRUNCATED_POWER_LAW = 'truncated_power_law'
POWER_LAW = 'power_law'
LOG_NORMAL = 'lognormal'
EXPONENTIAL = 'exponential'

# STATUSes

SUCCESS = 'success'
FAILED = 'failed'
OVER_TRAINED = 'over-trained'
UNDER_TRAINED = 'under-trained'

UNKNOWN = 'unknown'

GLOROT_FIX = 'glorot_fix'
NORMALIZE = 'normalize'

LAYERS = 'layers'

SAVEFIG = 'savefig'
SAVEDIR = 'savedir'

DELTA_ES = 'deltaEs'
INTRA = 'intra'
CONV2D_FFT = 'conv2d_fft'
CONV2D_NORM = 'conv2d_norm'

GLOROT_FIT = 'glorot_fit'

WW2X = 'ww2x'
VECTORS = 'vectors'
SMOOTH = 'smooth'
SVD_METHOD = 'svd_method'
FIX_FINGERS = 'fix_fingers'
MP_FIT = 'mp_fit'
FIT = 'fit'

RESCALE = 'rescale'
RANDOMIZE = 'randomize'
SPARSIFY = 'sparsify'
DETX = 'detX' # compute detx and for smoothing
LAMBDA_MIN = 'alpha_min' # smoothing

MIN_EVALS = 'min_evals'
DEFAULT_MIN_EVALS = 50
MIN_NUM_EVALS = 10

MAX_EVALS = 'max_evals'
DEFAULT_MAX_EVALS = 10000

PLOT = 'plot'
STACKED = 'stacked'

CHANNELS_STR = 'channels'
FIRST = 'first'
LAST = 'last'   

TOLERANCE = 'tolerance'
WEAK_RANK_LOSS_TOLERANCE = 0.000001 # on ei=gen values
    
# These are NOT the defaults...see analyze() for actual defaults
DEFAULT_PARAMS = {GLOROT_FIX: False, NORMALIZE:False, CONV2D_NORM:True, RANDOMIZE: True, 
                  SAVEDIR:DEF_SAVE_DIR, SAVEFIG:True, RESCALE:True, PLOT:False,
                  DELTA_ES:False, INTRA:False, CHANNELS_STR:None, CONV2D_FFT:False, 
                  WW2X:False, VECTORS:True, SMOOTH:None, STACKED:False, 
                  SVD_METHOD:FULL_SVD,  FIX_FINGERS:None, FIT:POWER_LAW, 
                  SPARSIFY: True, DETX: True,  MP_FIT:False,
                  MIN_EVALS:DEFAULT_MIN_EVALS, MAX_EVALS:DEFAULT_MAX_EVALS, 
                  TOLERANCE:WEAK_RANK_LOSS_TOLERANCE}


EVALS_THRESH = 0.00001

class LAYER_TYPE(IntFlag):
    UNKNOWN = auto()
    STACKED = auto()
    DENSE = auto()
    CONV1D = auto()
    CONV2D = auto()
    FLATTENED = auto()
    EMBEDDING = auto()
    NORM = auto()

    
class FRAMEWORK(IntFlag):
    UNKNOWN = auto()
    PYTORCH = auto()
    KERAS = auto()
    ONNX = auto()
    

class CHANNELS(IntFlag):
    UNKNOWN = auto()
    FIRST = auto()
    LAST = auto()   
            
    
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


# XMAX, XMIN not really used
    
class XMAX(IntFlag):
    UNKNOWN = auto()
    AUTO = auto()


class XMIN(IntFlag):
    UNKNOWN = auto()
    AUTO = auto()
