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


MAX_NUM_EVALS = 50000
DEF_SAVE_DIR = 'ww-img'


TRUNCATED_SVD = 'truncated_svd'
FULL_SVD = 'full_svd'

# fi_ fingers options
XMIN_PEAK = 'xmin_peak'
CLIP_XMAX = 'clip_xmax'

SVD = 'svd' # TruncatedSVD for Smoothing
RMT = 'rmt' # pyRMT / RIE
    
DEFAULT_PARAMS = {'glorot_fix': False, 'normalize':False, 'conv2d_norm':True, 'randomize': True, 
                  'savedir':DEF_SAVE_DIR, 'savefig':True, 'rescale':True, 'plot':False,
                  'deltaEs':False, 'intra':False, 'channels':None, 'conv2d_fft':False, 
                  'ww2x':False, 'vectors':False, 'smooth':None, 'stacked':False, 
                  'svd_method':FULL_SVD,  'fix_fingers':None}


TPL = 'truncated_power_law'
POWER_LAW = 'power_law'
LOG_NORMAL = 'lognormal'
EXPONENTIAL = 'exponential'

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
    
    
class PLOT(IntFlag):
    POWERLAW = auto()
    ESD = auto()
    ESDLOG = auto()
    
    
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
