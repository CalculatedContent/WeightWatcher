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
#
import sys, os
import logging

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import powerlaw
 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import torch
import torch.nn as nn

#
# this is use to allow editing in Eclipse but also
# building on the commend line
# see: https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
#
if __package__ is None or __package__ == '':
    # uses current directory visibility
    from RMT_Util import *
    from constants import *
else:
    # uses current package visibility
    from .RMT_Util import *
    from .constants import *

# TODO:  allow configuring custom logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('weightwatcher')  # ww.__name__

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

MAX_NUM_EVALS = 1000

DEFAULT_PARAMS = {'glorot_fix': False, 'normalize':False, 'conv2d_norm':True, 'randomize': True}
    

def main():
    """
    Weight Watcher
    """
    print("WeightWatcher command line support coming later. https://calculationconsulting.com")


class WWLayer:
    """WW wrapper layer to Keras and PyTorch Layer layer objects
       Uses pythong metaprogramming to add result columns for the final details dataframe"""
       
    def __init__(self, layer, layer_id=-1, name=None,
                 the_type=LAYER_TYPE.UNKNOWN, framework=FRAMEWORK.UNKNOWN, skipped=False):
        self.layer = layer
        self.layer_id = layer_id  
        self.name = name
        self.skipped = skipped
        self.the_type = the_type
        self.framework = framework
        
        self.channels = CHANNELS.UNKNOWN

        if (self.framework == FRAMEWORK.KERAS):
            self.channels = CHANNELS.FIRST
        elif (self.framework == FRAMEWORK.PYTORCH):
            self.channels = CHANNELS.LAST
        
        # get the LAYER_TYPE
        self.the_type = self.layer_type(self.layer)
        
        if self.name is None and hasattr(self.layer, 'name'):
            name = self.layer.name

        # original weights (tensor) and biases
        self.has_weights = False
        self.weights = None
  
        # extracted weight matrices
        self.num_W = 0
        self.Wmats = []

        self.N = 0
        self.M = 0
        self.num_components = self.M  # default for full SVD, not used yet
        self.rf = 1  # receptive field size, default for dense layer
        self.conv2d_count = 1  # reset by slice iterator for back compatability with ww2x
        self.w_norm = 1 # reset if normalize, conv2D_norm, or glorot_fix used

        # to be used for conv2d_fft approach
        self.inputs_shape = []
        self.outputs_shape = []
        
        # evals 
        self.evals = None
        self.rand_evals = None
        
        # details, set by metaprogramming in apply_xxx() methods
        self.columns = []
        self.make_weights()
        
    def add_column(self, name, value):
        """Add column to the details dataframe"""
        self.columns.append(name)
        self.__dict__[name] = value
        
    def get_row(self):
        """get a details dataframe row from the columns and metadata"""
        data = {}
        
        data['layer_id'] = self.layer_id
        data['name'] = self.name
        data['layer_type'] = str(self.the_type)
        data['N'] = self.N
        data['M'] = self.M
        data['rf'] = self.rf
        
        for col in self.columns:
            data[col] = self.__dict__[col]
                    
        return data
    
    def layer_type(self, layer):
        """Given a framework layer, determine the weightwatcher LAYER_TYPE
        This can detect basic Keras and PyTorch classes by type, and will try to infer the type otherwise. """

        the_type = LAYER_TYPE.UNKNOWN
       
        typestr = (str(type(layer))).lower()
            
        # Keras TF 2.x types
        if isinstance(layer, keras.layers.Dense): 
            the_type = LAYER_TYPE.DENSE
            
        elif isinstance(layer, keras.layers.Conv1D):                
            the_type = LAYER_TYPE.CONV1D
        
        elif isinstance(layer, keras.layers.Conv2D):                
            the_type = LAYER_TYPE.CONV2D
            
        elif isinstance(layer, keras.layers.Flatten):
            the_type = LAYER_TYPE.FLATTENED
            
        elif isinstance(layer, keras.layers.Embedding):
            the_type = LAYER_TYPE.EMBEDDING
            
        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            the_type = LAYER_TYPE.NORM
        
        # PyTorch        
             
        elif isinstance(layer, nn.Linear):
            the_type = LAYER_TYPE.DENSE
            
        elif isinstance(layer, nn.Conv1d):
            the_type = LAYER_TYPE.CONV1D
        
        elif isinstance(layer, nn.Conv2d):
            the_type = LAYER_TYPE.CONV2D
            
        elif isinstance(layer, nn.Embedding):
            the_type = LAYER_TYPE.EMBEDDING
                
        elif isinstance(layer, nn.LayerNorm):
            the_type = LAYER_TYPE.NORM
        
        # try to infer type (i.e for huggingface)
        elif typestr.endswith(".linear'>"):
            the_type = LAYER_TYPE.DENSE
            
        elif typestr.endswith(".dense'>"):
            the_type = LAYER_TYPE.DENSE
            
        elif typestr.endswith(".conv1d'>"):
            the_type = LAYER_TYPE.CONV1D
            
        elif typestr.endswith(".conv2d'>"):
            the_type = LAYER_TYPE.CONV2D
        
        return the_type
    
    def make_weights(self):
        """ Constructor for WWLayer class.  Make a ww (wrapper)_layer from a framework layer, or return None if layer is skipped.
        In particular , late uses specify filter on layer ids and names """
        
        has_weights = False;
        if not self.skipped:
            has_weights, weights, has_biases, biases = self.get_weights_and_biases()
            
            self.has_weights = has_weights
            self.has_biases = has_biases
            
            if has_biases:
                self.biases = biases   
                
            if has_weights:    
                self.weights = weights
                self.set_weight_matrices(weights)
    
        return self
        
    def get_weights_and_biases(self):
        """extract the original weights (as a tensor) for the layer, and biases for the layer, if present"""
        
        has_weights, has_biases = False, False
        weights, biases = None, None
    
        if self.framework == FRAMEWORK.PYTORCH:
            if hasattr(self.layer, 'weight'):
                w = [np.array(self.layer.weight.data.clone().cpu())]
                has_weights = True
                
        elif self.framework == FRAMEWORK.KERAS:
            w = self.layer.get_weights()
            if(len(w) > 0):
                has_weights = True
                
            if(len(w) > 1):
                has_biases = True
                
        else:
            logger.error("unknown framework: weighwatcher only supports keras (tf 2.x) or pytorch ")
       
        if has_weights:
            if len(w) == 1:
                logger.debug("Linear weights shape  len(w){} type(w){}  w.shape {} ".format(len(w), type(w), w[0].shape))
                weights = w[0]
                biases = None
            elif len(w) == 2:
                weights = w[0]
                biases = w[1]
            else:
                logger.error("unknown weights, with len(w)={} ".format(len(w)))
        
        return has_weights, weights, has_biases, biases  
      
    def set_weight_matrices(self, weights, conv2d_fft=False, conv2d_norm=True):
        """extract the weight matrices from the framework layer weights (tensors)
        sets the weights and detailed properties on the ww (wrapper) layer 
    
        conv2d_fft not supported yet """
   
        if not self.has_weights:
            logger.info("Layer {} {} has no weights".format(self.layer_id, self.name))
            return 
        
        the_type = self.the_type
        
        N, M, n_comp, rf = 0, 0, 0, None
        Wmats = []
        
        # this may change if we treat Conv1D differently layer
        if (the_type == LAYER_TYPE.DENSE or the_type == LAYER_TYPE.CONV1D):
            Wmats = [self.weights]
            N, M = np.max(Wmats[0].shape), np.min(Wmats[0].shape)
            n_comp = M
            rf = 1
            
        # TODO: reset channels nere ?    
        elif the_type == LAYER_TYPE.CONV2D:
            Wmats, N, M, rf, channels = self.conv2D_Wmats(weights)
            n_comp = M
            self.channels = channels
            
        elif the_type == LAYER_TYPE.NORM:
            logger.info("Layer id {}  Layer norm has no matrices".format(self.layer_id))
        
        else:
            logger.info("Layer id {}  unknown type {} layer  {}".format(self.layer_id, the_type, self.layer))
    
        self.N = N
        self.M = M
        self.rf = rf
        self.Wmats = Wmats
        self.num_components = n_comp
        
        return 
        
    def __repr__(self):
        return "WWLayer()"

    def __str__(self):
        return "WWLayer {}  {} {} {}  skipped {}".format(self.layer_id, self.name,
                                                       self.framework.name, self.the_type.name, self.skipped)
    
    def conv2D_Wmats(self, Wtensor, channels=CHANNELS.UNKNOWN):
        """Extract W slices from a 4 layer_id conv2D tensor of shape: (N,M,i,j) or (M,N,i,j).  
        Return ij (N x M) matrices, with receptive field size (rf) and channels flag (first or last)"""
        
        logger.debug("conv2D_Wmats")
        
        # TODO:  detect or use channels
        # if channels specified ...
    
        Wmats = []
        s = Wtensor.shape
        N, M, imax, jmax = s[0], s[1], s[2], s[3]
        if N + M >= imax + jmax:
            logger.debug("Channels Last tensor shape detected: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))
            
            channels = CHANNELS.LAST
            for i in range(imax):
                for j in range(jmax):
                    W = Wtensor[:, :, i, j]
                    if N < M:
                        W = W.T
                    Wmats.append(W)
        else:
            N, M, imax, jmax = imax, jmax, N, M          
            logger.debug("Channels First shape detected: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))
            
            channels = CHANNELS.FIRST
            for i in range(imax):
                for j in range(jmax):
                    W = Wtensor[i, j, :, :]
                    if N < M:
                        W = W.T
                    Wmats.append(W)
                    
        rf = imax * jmax  # receptive field size             
        logger.debug("get_conv2D_Wmats N={} M={} rf= {} channels = {}".format(N, M, rf, channels))
    
        return Wmats, N, M, rf, channels    


class ModelIterator:
    """Iterator that loops over ww wrapper layers, with original matrices (tensors) and biases (optional) available."""

    def __init__(self, model, params=DEFAULT_PARAMS):
        
        self.params = params
        self.k = 0
        
        self.model = model
        self.model_iter, self.framework = self.model_iter_(model) 
        
        self.layer_iter = self.make_layer_iter_()            
        
    def __iter__(self):
        return self
    
    # Python 3 compatibility
    def __next__(self):
        return self.next()
    
    def next(self):
        curr_layer = next(self.layer_iter)
        if curr_layer:    
            return curr_layer
        else:
            raise StopIteration()
    
    def model_iter_(self, model):
        """Return a generator for iterating over the layers in the model.  
        Also detects the framework being used. 
        Used by base class and child classes to iterate over the framework layers """
        layer_iter = None
        
        if hasattr(model, 'layers'):

            def layer_iter_():
                for layer in model.layers:
                        yield layer 
                        
            layer_iter = layer_iter_()
            framework = FRAMEWORK.KERAS
        elif hasattr(model, 'modules'):

            def layer_iter_():
                for layer in model.modules():
                        yield layer 
                        
            layer_iter = layer_iter_()    
            framework = FRAMEWORK.PYTORCH
        else:
            layer_iter = None
            framework = FRAMEWORK.UNKNOWN
            
        return layer_iter, framework
                      
    def make_layer_iter_(self):
        """The layer iterator for this class / instance.
         Override this method to change the type of iterator used by the child class"""
        return self.model_iter


class WWLayerIterator(ModelIterator):
    """Creates an iterator that generates WWLayer wrapper objects to the model layers"""

    def __init__(self, model, params=DEFAULT_PARAMS, filters=[]):
        
        super().__init__(model, params=params)
        
        self.filter_ids = []
        self.filter_types = []
        self.filter_names = []
        
        if type(filters) is not list:
            filters = [filters]
            
        for f in filters:
            tf = type(f)
        
            if tf is LAYER_TYPE:
                logger.info("Filtering layer by type {}".format(str(f)))
                self.filter_types.append(f)
            elif tf is int:
                logger.info("Filtering layer by id {}".format(f))
                self.filter_ids.append(f) 
            elif tf is str:
                logger.info("Filtering layer by name {}".format(f))
                self.filter_names.append(f) 
            else:
                logger.warn("unknown filter type {} detected and ignored".format(tf))
                
    def apply_filters(self, ww_layer):
        """Apply filters.  Set skipped False  if filter is applied to this layer, keeping the layer (or no filters, meaning all layers kept)"""
        ww_layer.skipped = False
          
        if self.filter_types is not None and len(self.filter_types) > 0:
            if ww_layer.the_type in self.filter_types:
                logger.info("keeping layer {} {} with type {} ".format(ww_layer.layer_id, ww_layer.name , str(ww_layer.the_type)))
                ww_layer.skipped = False
            else:
                logger.info("skipping layer {} {} with type {} ".format(ww_layer.layer_id, ww_layer.name , str(ww_layer.the_type)))
                ww_layer.skipped = True

                
        if self.filter_ids is not None and len(self.filter_ids) > 0:
            if ww_layer.layer_id in self.filter_ids:
                logger.info("keeping layer {} {} by id".format(ww_layer.layer_id, ww_layer.name))
                ww_layer.skipped = False
            else:
                logger.info("skipping layer {} {} by id".format(ww_layer.layer_id, ww_layer.name))
                ww_layer.skipped = True


                
        if self.filter_names is not None and len(self.filter_names) > 0:
            if ww_layer.name in self.filter_names:
                logger.info("keeping layer {} {} by name ".format(ww_layer.layer_id, ww_layer.name))
                ww_layer.skipped = False
            else:
                logger.info("skipping layer {} {} by name ".format(ww_layer.layer_id, ww_layer.name))
                ww_layer.skipped = True
     
        return ww_layer.skipped
    
    def ww_layer_iter_(self):
        """Create a generator for iterating over ww_layers, created lazily """
        for curr_layer in self.model_iter:
            curr_id, self.k = self.k, self.k + 1
            
            ww_layer = WWLayer(curr_layer, layer_id=curr_id, framework=self.framework)
            
            self.apply_filters(ww_layer)
            
            if not self.layer_supported(ww_layer):
                ww_layer.skipped = True
                        
            if not ww_layer.skipped:
                yield ww_layer    
                
    def make_layer_iter_(self):
        return self.ww_layer_iter_()
    
    def layer_supported(self, ww_layer):
        """Return true if this kind of layer is supported"""
        
        supported = False

        layer_id = ww_layer.layer_id
        name = ww_layer.name
        the_type = ww_layer.the_type
        rf = ww_layer.rf
        
        M = ww_layer.M
        N = ww_layer.N
        
        min_evals = self.params.get('min_evals')
        max_evals = self.params.get('max_evals')
        
        logger.debug("layer_supported  N {} max evals {}".format(N, max_evals))
        
        if ww_layer.skipped:
            logger.debug("Layer {} {} is skipped".format(layer_id, name))
            
        elif not ww_layer.has_weights:
            logger.debug("layer not supported: Layer {} {} has no weights".format(layer_id, name))
            return False
        
        elif the_type is LAYER_TYPE.UNKNOWN:
            logger.debug("layer not supported: Layer {} {} type {} unknown".format(layer_id, name, the_type))
            return False
        
        elif the_type in [LAYER_TYPE.FLATTENED, LAYER_TYPE.NORM]:
            logger.debug("layer not supported: Layer {} {} type {} not supported".format(layer_id, name, the_type))
            return False
        
        elif min_evals and M * rf <= min_evals:
            logger.debug("layer not supported: Layer {} {}: num_evals {} <  min_evals {}".format(layer_id, name, M * rf, min_evals))
            return False
                  
        elif max_evals and N * rf >= max_evals:
            logger.debug("layer not supported: Layer {} {}: num_evals {} > max_evals {}".format(layer_id, name, N * rf, max_evals))
            return False
        
        elif the_type in [LAYER_TYPE.DENSE, LAYER_TYPE.CONV1D, LAYER_TYPE.CONV2D]:
            supported = True
                        
        return supported
    

class WW2xSliceIterator(WWLayerIterator):
    """Iterator variant that breaks Conv2D layers into slices for back compatability"""
    from copy import deepcopy

    def ww_slice_iter_(self):
        
        for ww_layer in self.ww_layer_iter_():
            if ww_layer.the_type == LAYER_TYPE.CONV2D:
                
                count = len(ww_layer.Wmats)
                for iw, W in enumerate(ww_layer.Wmats):
                    ww_slice = deepcopy(ww_layer)
                    ww_slice.Wmats = [W]
                    ww_slice.conv2d_count = count
                    ww_slice.add_column("slice_id", iw)
                    yield ww_slice

            else:
                ww_layer.add_column("slice_id", 0)
                yield ww_layer
                
    def make_layer_iter_(self):
        return self.ww_slice_iter_()
    
    
class WeightWatcher(object):

    def __init__(self, model=None, log=True):
        self.model = self.load_model(model)
        self.details = None
        # self.setup_custom_logger(log, logger)     
        logger.info(self.banner())

#     def setup_custom_logger(self, log, logger):
#         formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
#     
#         handler = logging.StreamHandler()
#         handler.setFormatter(formatter)
#     
#         if not logger:
#            logger = logging.getLogger(__name__)
#         
#         if not logger.handlers: # do not register handlers more than once
#             if log:
#                 logging.setLevel(logging.INFO) 
#                 console_handler = logging.StreamHandler()
#                 formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
#                 console_handler.setFormatter(formatter)
#                 self.logger.addHandler(console_handler)
#             else:
#                 self.logger.addHandler(logging.NullHandler())
#   
#         return logger

    def header(self):
        """WeightWatcher v0.1.dev0 by Calculation Consulting"""
#        from weightwatcher import __name__, __version__, __author__, __description__, __url__
#        return "{} v{} by {}\n{}\n{}".format(__name__, __version__, __author__, __description__, __url__)
        return ""

    def banner(self):
        versions = "\npython      version {}".format(sys.version)
        versions += "\nnumpy       version {}".format(np.__version__)
        versions += "\ntensforflow version {}".format(tf.__version__)
        versions += "\nkeras       version {}".format(tf.keras.__version__)
        return "\n{}{}".format(self.header(), versions)

    def __repr__(self):
        done = bool(self.results)
        txt = "\nAnalysis done: {}".format(done)
        return "{}{}".format(self.header(), txt)
            
    # TODO: get rid of this or extend to be more generally useful
    def load_model(self, model):
        """load the model from a file, only works for keras right now"""
        res = model
        if isinstance(model, str):
            if os.path.isfile(model):
                logger.info("Loading model from file '{}'".format(model))
                res = load_model(model)
            else:
                logger.error("Loading model from file '{}': file not found".format(model))
        return res
    
    # TODO: implement
    def same_models(self, model_1, model_2):
        """Compare models to see if the are the same architecture.
        Not really impelemnted yet"""
    
        same = True
        layer_iter_1 = WWLayerIterator(model_1)
        layer_iter_2 = WWLayerIterator(model_2)
        
        same = layer_iter_1.framework == layer_iter_2.framework 

        return same
    
    def distances(self, model_1, model_2):
        """Compute the distances between model_1 and model_2 for each layer. 
        Reports Frobenius norm of the distance between each layer weights (tensor)
        
           < ||W_1-W_2|| >
           
        output: avg delta W, a details dataframe
           
        models should be the same size and from the same framework
           
        """
        
        # check and throw exception if inputs incorrect
        # TODO: review design here...may need something else
        #   need to:
        # .   - iterate over all layers and check
        # .   - inspect framework by framework
        # .   - check here instead
        #
        
        same = True
        layer_iter_1 = WWLayerIterator(model_1)
        layer_iter_2 = WWLayerIterator(model_2)
        
        same = layer_iter_1.framework == layer_iter_2.framework 
        if not same:
            raise Exception("Sorry, models are from different frameworks")
        
        details = pd.DataFrame(columns=['layer_id', 'name', 'delta_W', 'delta_b', 'W_shape', 'b_shape'])
        data = {}
        
        try:      
            for layer_1, layer_2 in zip(layer_iter_1, layer_iter_2):
                data['layer_id'] = layer_1.layer_id
                data['name'] = layer_1.name
    
                if layer_1.has_weights:
                    data['delta_W'] = np.linalg.norm(layer_1.weights - layer_2.weights)
                    data['W_shape'] = layer_1.weights.shape
    
                    if layer_1.has_biases:
                        data['delta_b'] = np.linalg.norm(layer_1.biases - layer_2.biases)
                        data['b_shape'] = layer_1.biases.shape
    
                    details = details.append(data, ignore_index=True)
        except:
            logger.error("Sorry, problem comparing models")
            raise Exception("Sorry, problem comparing models")
        
        details.set_layer_id('layer_id', inplace=True)
        avg_dW = np.mean(details['delta_W'].to_numpy())
        return avg_dW, details
    
    def combined_eigenvalues(self, Wmats, N, M, n_comp, params):
        """Compute the eigenvalues for all weights of the NxM weight matrices (N >= M), 
            combined into a single, sorted, numpy array
    
            Applied normalization and glorot_fix if specified
    
            Assumes an array of weights comes from a conv2D layer and applies conv2d_norm normalization by default
    
            Also returns max singular value and rank_loss, needed for other calculations
         """
    
        all_evals = []
        max_sv = 0.0
        rank_loss = 0
    
        # TODO:  allow user to specify
        normalize = params['normalize']
        glorot_fix = params['glorot_fix']
        conv2d_norm = params['conv2d_norm']  # True
        
        if type(Wmats) is not list:
            logger.debug("combined_eigenvalues: Wmats -> [WMmats]")
            Wmats = [Wmats]
    
        count = len(Wmats)
        for  W in Wmats:
    
            Q = N / M  
            # SVD can be swapped out here
            # svd = TruncatedSVD(n_components=M-1, n_iter=7, random_state=10)
    
            W = W.astype(float)
            logger.debug("Running full SVD:  W.shape={}  n_comp = {}".format(W.shape, n_comp))
            sv = np.linalg.svd(W, compute_uv=False)
            sv = sv.flatten()
            sv = np.sort(sv)[-n_comp:]
            # TODO:  move to PL fit for robust estimator
            # if len(sv) > max_evals:
            #    #logger.info("chosing {} singular values from {} ".format(max_evals, len(sv)))
            #    sv = np.random.choice(sv, size=max_evals)
    
            # sv = svd.singular_values_
            evals = sv * sv
            if normalize:
                evals = evals / N
    
            all_evals.extend(evals)
    
            max_sv = np.max([max_sv, np.max(sv)])
            max_ev = np.max(evals)
            rank_loss = 0  # rank_loss + self.calc_rank_loss(sv, M, max_ev)
    
        return np.sort(np.array(all_evals)), max_sv, rank_loss
            
            
    def apply_normalize_Wmats(self, ww_layer, params=DEFAULT_PARAMS):
        """Normalize the W matrix or Wmats """

        normalize = params['normalize']
        glorot_fix = params['glorot_fix']
        conv2d_norm = params['conv2d_norm']
        
        M = ww_layer.M
        N = ww_layer.N
        rf = ww_layer.rf
        norm = ww_layer.w_norm # shoud be 1.0 unless reset for some reason
        
        Wmats = ww_layer.Wmats
        new_Wmats = []
        
        if type(Wmats) is not list:
            logger.debug("combined_eigenvalues: Wmats -> [WMmats]")
            Wmats = [Wmats]
               
        for  W in Wmats:
            # not really used
            rf_size = ww_layer.conv2d_count
            check, checkTF = self.glorot_norm_check(W, N, M, rf_size) 
            
            if glorot_fix:
                norm = self.glorot_norm_fix(W, N, M, rf_size)
                
            elif conv2d_norm and ww_layer.the_type is LAYER_TYPE.CONV2D:
                # w_norm is reset in slices to fix this
                norm = np.sqrt(ww_layer.conv2d_count/2.0)
                
            if normalize and not glorot_fix:
                norm = 1 / np.sqrt(N)
               
            W = W * norm
            ww_layer.w_norm = norm
            new_Wmats.append(W)
    
        ww_layer.Wmats = new_Wmats
        return ww_layer
                
        
                 
    def apply_esd(self, ww_layer, params=DEFAULT_PARAMS):
        """run full SVD on layer weight matrices, compute ESD on combined eigenvalues, combine all,  and save to layer """
        
        layer_id = ww_layer.layer_id
        name = ww_layer.name
        the_type = ww_layer.the_type
         
        M = ww_layer.M
        N = ww_layer.N
        rf = ww_layer.rf
        
        logger.debug("apply ESD  on Layer {} {} ".format(layer_id, name))
                        
        logger.debug("running SVD on Layer {} {} ".format(layer_id, name))
        logger.debug("params {} ".format(params))
    
        Wmats = ww_layer.Wmats
        n_comp = ww_layer.num_components
                
        evals, sv_max, rank_loss = self.combined_eigenvalues(Wmats, N, M, n_comp, params)
     
        ww_layer.evals = evals
        ww_layer.add_column("has_esd", True)
        ww_layer.add_column("num_evals", len(evals))
        ww_layer.add_column("sv_max", sv_max)
        ww_layer.add_column("rank_loss", rank_loss)
        ww_layer.add_column("lambda_max", np.max(evals))
            
        return ww_layer
    
    def apply_random_esd(self, ww_layer, params=DEFAULT_PARAMS):
        """Randomize the layer weight matrices, compute ESD on combined eigenvalues, combine all,  and save to layer """
        
        layer_id = ww_layer.layer_id
        name = ww_layer.name
        the_type = ww_layer.the_type
         
        M = ww_layer.M
        N = ww_layer.N
        rf = ww_layer.rf
        
        logger.debug("apply random ESD  on Layer {} {} ".format(layer_id, name))
                        
        logger.debug("running SVD on Layer {} {} ".format(layer_id, name))
        logger.debug("params {} ".format(params))
    
        Wmats = ww_layer.Wmats
        n_comp = ww_layer.num_components
        num_replicas = 1
        # hack to improve random estimator if we don't have that many evals
        if n_comp < 100:
            num_replicas = 5
        
        rand_evals = self.random_eigenvalues(Wmats, n_comp, num_replicas , params)
     
        ww_layer.rand_evals = rand_evals
        ww_layer.add_column("max_rand_eval", np.max(rand_evals))
        
        if params['plot']:
            self.plot_random_esd(ww_layer, params)
            
        return ww_layer
    
           
    def apply_plot_esd(self, ww_layer, params=DEFAULT_PARAMS):
        """Plot the ESD on regular and log scale.  Only used when powerlaw fit not called"""
                    
        evals = ww_layer.evals
        name = ww_layer.name
        
        plt.title(name)
        plt.hist(evals, bins=100)
        plt.show()
        
        plt.title(name)
        plt.hist(np.log10(evals), bins=100)
        plt.show()
            
        return ww_layer
    
    
 
    def apply_fit_powerlaw(self, ww_layer, params=DEFAULT_PARAMS):
        """Plot the ESD on regular and log scale.  Only used when powerlaw fit not called"""
                    
        evals = ww_layer.evals
        layer_id = ww_layer.layer_id
        name = ww_layer.name
        title = "{} {}".format(layer_id, name)

        xmin = None  # TODO: allow other xmin settings
        xmax = np.max(evals)
        plot = params['plot']
        sample = False  # TODO:  decide if we want sampling for large evals       
        sample_size = None
              
        alpha, xmin, xmax, D, sigma, num_pl_spikes = self.fit_powerlaw(evals, xmin=xmin, xmax=xmax, plot=plot, title="", sample=sample, sample_size=sample_size)
        
        ww_layer.add_column('alpha', alpha)
        ww_layer.add_column('xmin', xmin)
        ww_layer.add_column('xmax', xmax)
        ww_layer.add_column('D', D)
        ww_layer.add_column('sigma', sigma)
        ww_layer.add_column('num_pl_spikes', num_pl_spikes)

        return ww_layer

    # test with https://github.com/osmr/imgclsmob/blob/master/README.md
    def analyze(self, model=None, layers=[], min_evals=0, max_evals=None,
                min_size=None, max_size=None,  # deprecated
                normalize=False, glorot_fix=False, plot=False, randomize=False, 
                mp_fit=False, conv2d_fft=False,conv2d_norm=True, fit_bulk=False, ww2x=False):#, params=DEFAULT_PARAMS):
        """
        Analyze the weight matrices of a model.

        layers:
            List of layer ids. If empty, analyze all layers (default)
        min_evals:
            Minimum number of evals (M*rf) 
        max_evals:
            Maximum number of evals (N*rf) (0 = no limit)
        normalize:
            Normalize the X matrix. Usually True for Keras, False for PyTorch.
            Ignored if glorot_norm is set
        glorot_fix:
            Adjust the norm for the Glorot Normalization.  
        alphas:
            # deprecated
            Compute the power laws (alpha) of the weight matrices. 
            Time consuming so disabled by default (use lognorm if you want speed)
        lognorms:
            # deprecated
            Compute the log norms of the weight matrices.
        spectralnorms:
            # deprecated
            Compute the spectral norm (max eigenvalue) of the weight matrices.
        softranks:
            # deprecated
            Compute the soft norm (i.e. StableRank) of the weight matrices.
        mp_fit:
            Compute the best Marchenko-Pastur fit of each weight matrix ESD
        conv2d_fft:
            For Conv2D layers, use FFT method.  Otherwise, extract and combine the weight matrices for each receptive field
            Note:  for conf2d_fft, the ESD is automatically subsampled to max_evals eigenvalues max
        fit_bulk: 
            Attempt to fit bulk region of ESD only  N/A yet
        ww2x:
            Use weightwatcher version 0.2x style iterator, which slices up Conv2D layers in N=rf matrices
        device: N/A yet
            if 'gpu'  use torch.svd()
            else 'cpu' use np.linalg.svd
        params:  
            N/A as inputs: dictionary of default parameters, which can be set but will be over-written by 
        """

        model = model or self.model   
        
        if min_size or max_size:
            logger.warn("min_size and max_size options changed to min_evals, max_evals, ignored for now")     
        
        # I need to figure this out
        # can not specify params on input yet
        # maybe just have a different analyze() that only uses this 
        
        params=DEFAULT_PARAMS
        params['min_evals'] = min_evals 
        params['max_evals'] = max_evals
        params['plot'] = plot
        params['randomize'] = randomize
        params['mp_fit'] = mp_fit
        params['normalize'] = normalize
        params['glorot_fix'] = glorot_fix
        params['conv2d_norm'] = conv2d_norm
        params['ww2x'] = ww2x

            
        logger.info("params {}".format(params))
        if not self.valid_params(params):
            logger.error("Error, params not valid: \n {}".format(params))
   
        if ww2x:
            logger.info("Using weightwatcher 0.2x style layer and slice iterator")
            layer_iterator = WW2xSliceIterator(model, filters=layers, params=params)     
        else:
            layer_iterator = WWLayerIterator(model, filters=layers, params=params)     
        
        details = pd.DataFrame(columns=['layer_id', 'name'])
           
        for ww_layer in layer_iterator:
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.info("LAYER: {} {}  : {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.layer)))
                
                self.apply_normalize_Wmats(ww_layer, params)
                self.apply_esd(ww_layer, params)
                
                
                if ww_layer.evals is not None:
                    self.apply_fit_powerlaw(ww_layer, params)
                    if params['mp_fit']:
                        logger.info("MP Fitting Layer: {} {} ".format(ww_layer.layer_id, ww_layer.name)) 
                        self.apply_mp_fit(ww_layer, random=False, params=params)
                    
                    if params['randomize'] or params['mp_fit']:
                        logger.info("Randomizing Layer: {} {} ".format(ww_layer.layer_id, ww_layer.name))
                        self.apply_random_esd(ww_layer, params)
                        logger.info("MP Fitting Random layer: {} {} ".format(ww_layer.layer_id, ww_layer.name)) 
                        self.apply_mp_fit(ww_layer, random=True, params=params)
                    
                    self.apply_norm_metrics(ww_layer, params)
                    
                # TODO: add find correlation traps here
                details = details.append(ww_layer.get_row(), ignore_index=True)

        self.details = details
        return details
    
    def get_details(self):
        """get the current details, created by analyze"""
        return self.details
    
    def get_summary(self, details=None):
        """Return metric averages, as dict, if available """
        
        summary = {}
        if details is None:
            details = self.details
        
        columns = []  
        if details is not None:
             columns = details.columns
            
        metrics = ["log_norm","alpha","alpha_weighted","log_alpha_norm", "log_spectral_norm","stable_rank","mp_softrank"]
        for metric in metrics:
            if metric in columns:
                summary[metric]=details[metric].mean()
                
        return summary

    
    # test with https://github.com/osmr/imgclsmob/blob/master/README.md
    def describe(self, model=None, layers=[], min_evals=0, max_evals=None,
                min_size=None, max_size=None,  # deprecated
                normalize=False, glorot_fix=False, plot=False, mp_fit=False, conv2d_fft=False,
                conv2d_norm=True, fit_bulk=False,  ww2x=False):
        """
        Same as analyze() , but does not run the ESD or Power law fits
        
        """

        model = model or self.model    
        
        if min_size or max_size:
            logger.warn("min_size and max_size options changed to min_evals, max_evals, ignored for now")     
        
        params = DEFAULT_PARAMS
        params['min_evals'] = min_evals
        params['max_evals'] = max_evals
        params['plot'] = plot
        params['normalize'] = normalize
        params['glorot_fix'] = glorot_fix
        params['conv2d_norm'] = conv2d_norm 
            
        logger.info("params {}".format(params))
        if not self.valid_params(params):
            logger.error("Error, params not valid: \n {}".format(params))
   
        if ww2x:
            logger.info("Using weightwatcher 0.2x style layer and slice iterator")
            layer_iterator = WW2xSliceIterator(model, filters=layers, params=params)     
        else:
            layer_iterator = WWLayerIterator(model, filters=layers, params=params)  
   
        
        details = pd.DataFrame(columns=['layer_id', 'name'])
           
        for ww_layer in layer_iterator:
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.debug("LAYER TYPE: {} {}  layer type {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.layer)))
                logger.debug("weights shape : {}  max size {}".format(ww_layer.weights.shape, params['max_evals']))
                ww_layer.add_column('num_evals', ww_layer.M * ww_layer.rf)
                details = details.append(ww_layer.get_row(), ignore_index=True)

        return details

    def valid_params(self, params):
        """Vlaidate the input parametersm, return True if valid, False otherwise"""
        
        valid = True
        
        xmin = params.get('xmin')
        if xmin and xmin not in [XMIN.UNKNOWN, XMIN.AUTO, XMIN.PEAK]:
            logger.warn("param xmin unknown, ignoring {}".format(xmin))
            valid = False
            
        xmax = params.get('xmax')
        if xmax and xmax not in [XMAX.UNKNOWN, XMIN.AUTO]:
            logger.warn("param xmax unknown, ignoring {}".format(xmax))
            valid = False
        
        min_evals = params.get('min_evals') 
        max_evals = params.get('max_evals')
        if min_evals and max_evals and min_evals >= max_evals:
            logger.warn("min_evals {} > max_evals {}".format(min_evals, max_evals))
            valid = False
        elif max_evals and max_evals < -1:
            logger.warn(" max_evals {} < -1 ".format(max_evals))
            valid = False
        
        return valid
    
#      # @deprecated
#     def print_results(self, results=None):
#         self.compute_details(results=results)
# 
#     # @deprecated
#     def get_details(self, results=None):
#         """
#         Deprecated: returns a pandas dataframe with details for each layer
#         """
#         return self.details
    
    
 # not used yet   
    def get_conv2D_fft(self, W, n=32):
        """Compute FFT of Conv2D channels, to apply SVD later"""
        
        logger.info("get_conv2D_fft on W {}".format(W.shape))

        # is pytorch or tensor style 
        s = W.shape
        logger.debug("    Conv2D SVD ({}): Analyzing ...".format(s))

        N, M, imax, jmax = s[0], s[1], s[2], s[3]
        # probably better just to check what col N is in 
        if N + M >= imax + jmax:
            logger.debug("[2,3] tensor shape detected: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))    
            fft_axes = [2, 3]
        else:
            N, M, imax, jmax = imax, jmax, N, M          
            fft_axes = [0, 1]
            logger.debug("[1,2] tensor shape detected: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))

        # Switch N, M if in wrong order
        if N < M:
            M, N = N, M

        #  receptive_field / kernel size
        rf = np.min([imax, jmax])
        # aspect ratio
        Q = N / M 
        # num non-zero eigenvalues  rf is receptive field size (sorry calculated again here)
        n_comp = rf * N * M
        
        logger.info("N={} M={} n_comp {} ".format(N, M, n_comp))

        # run FFT on each channel
        fft_grid = [n, n]
        fft_coefs = np.fft.fft2(W, fft_grid, axes=fft_axes)
        
        return [fft_coefs], N, M, n_comp

    # not used
    def normalize_evals(self, evals, N, M):
        """Normalizee evals matrix by 1/N"""
        logger.debug(" normalzing evals, N, M {},{},{}".format(N, M))
        return evals / N

    def glorot_norm_fix(self, W, N, M, rf_size):
        """Apply Glorot Normalization Fix """

        kappa = np.sqrt(2 / ((N + M) * rf_size))
        W = W / kappa
        return W , 1/kappa

    def pytorch_norm_fix(self, W, N, M, rf_size):
        """Apply pytorch Channel Normalization Fix

        see: https://chsasank.github.io/vision/_modules/torchvision/models/vgg.html
        """

        kappa = np.sqrt(2 / (N * rf_size))
        W = W / kappa
        return W 

    def glorot_norm_check(self, W, N, M, rf_size,
                   lower=0.5, upper=1.5):
        """Check if this layer needs Glorot Normalization Fix"""

        kappa = np.sqrt(2 / ((N + M) * rf_size))
        norm = np.linalg.norm(W)

        check1 = norm / np.sqrt(N * M)
        check2 = norm / (kappa * np.sqrt(N * M))
        
        if (rf_size > 1) and (check2 > lower) and (check2 < upper):   
            return check2, True
        elif (check1 > lower) & (check1 < upper): 
            return check1, True
        else:
            if rf_size > 1:
                return check2, False
            else:
                return check1, False
    
    def random_eigenvalues(self, Wmats, n_comp, num_replicas=1, params=DEFAULT_PARAMS):
        """Compute the eigenvalues for all weights of the NxM skipping layer, num evals ized weight matrices (N >= M), 
            combined into a single, sorted, numpy array.  
    
        see: combined_eigenvalues()
        
         """
         
        normalize = params['normalize']
        glorot_fix = params['glorot_fix']
        conv2d_norm = params['conv2d_norm']  # True
         
        all_evals = []

        logger.info("generating {} replicas for each W of the random eigenvalues".format(num_replicas))
        for num in range(num_replicas):
            count = len(Wmats)
            for  W in Wmats:
    
                M, N = np.min(W.shape), np.max(W.shape)
                Q = N / M
               
                Wrand = W.flatten()
                np.random.shuffle(Wrand)
                W = Wrand.reshape(W.shape)
                W = W.astype(float)
                logger.debug("Running Randomized Full SVD")
                sv = np.linalg.svd(W, compute_uv=False)
                sv = sv.flatten()
                sv = np.sort(sv)[-n_comp:]    
                
                # sv = svd.singular_values_
                evals = sv * sv 
                all_evals.extend(evals)
                                       
        return np.sort(np.array(all_evals))
   
    def plot_random_esd(self, ww_layer, params=DEFAULT_PARAMS):
        """Plot histogram and log histogram of ESD and randomized ESD"""
          
        evals = ww_layer.evals
        rand_evals = ww_layer.rand_evals
        title = "Layer {} {}: ESD & Random ESD".format(ww_layer.layer_id,ww_layer.name)
          
        nonzero_evals = evals[evals > 0.0]
        nonzero_rand_evals = rand_evals[rand_evals > 0.0]
        max_rand_eval = np.max(rand_evals)

        plt.hist((nonzero_evals), bins=100, density=True, color='g', label='original')
        plt.hist((nonzero_rand_evals), bins=100, density=True, color='r', label='random', alpha=0.5)
        plt.axvline(x=(max_rand_eval), color='orange', label='max rand')
        plt.title(title)   
        plt.xlabel(r" Eigenvalues $(\lambda)$")               
        plt.legend()
        plt.show()

        plt.hist(np.log10(nonzero_evals), bins=100, density=True, color='g', label='original')
        plt.hist(np.log10(nonzero_rand_evals), bins=100, density=True, color='r', label='random', alpha=0.5)
        plt.axvline(x=np.log10(max_rand_eval), color='orange', label='max rand')
        title = "Layer {} {}: Log10 ESD & Random ESD".format(ww_layer.layer_id,ww_layer.name)
        plt.title(title)   
        plt.xlabel(r"Log10 Eigenvalues $(log_{10}\lambda)$")               
        plt.legend()
        plt.show()
        
    # Mmybe should be static function    
    def calc_rank_loss(self, singular_values, M, lambda_max):
        """compute the rank loss for these singular given the tolerances
        """
        sv = singular_values
        tolerance = lambda_max * M * np.finfo(np.max(sv)).eps
        return np.count_nonzero(sv > tolerance, axis=-1)
            
    def fit_powerlaw(self, evals, xmin=None, xmax=None, plot=True, title="", sample=False, sample_size=None):
        """Fit eigenvalues to powerlaw
        
            if xmin is 
                'auto' or None, , automatically set this with powerlaw method
                'peak' , try to set by finding the peak of the ESD on a log scale
            
            if xmax is 'auto' or None, xmax = np.max(evals)
                     
         """
             
        num_evals = len(evals)
        logger.debug("fitting power law on {} eigenvalues".format(num_evals))
        
        # TODO: replace this with a robust sampler / stimator
        # requires a lot of refactoring below
        if sample and  sample_size is None:
            logger.info("setting sample size to default MAX_NUM_EVALS={}".format(MAX_NUM_EVALS))
            sample_size = MAX_NUM_EVALS
            
        if sample and num_evals > sample_size:
            logger.warn("samping not implemented in production yet")
            logger.info("chosing {} eigenvalues from {} ".format(sample_size, len(evals)))
            evals = np.random.choice(evals, size=sample_size)
                    
        if xmax == XMAX.AUTO or xmax is XMAX.UNKNOWN or xmax is None:
            xmax = np.max(evals)
            
        if xmin == XMAX.AUTO  or xmin is None:
            fit = powerlaw.Fit(evals, xmax=xmax, verbose=False)
        elif xmin == XMAX.PEAK :
            nz_evals = evals[evals > 0.0]
            num_bins = 100  # np.min([100, len(nz_evals)])
            h = np.histogram(np.log10(nz_evals), bins=num_bins)
            ih = np.argmax(h[0])
            xmin2 = 10 ** h[1][ih]
            xmin_range = (0.95 * xmin2, 1.05 * xmin2)
            fit = powerlaw.Fit(evals, xmin=xmin_range, xmax=xmax, verbose=False)   
        else:
            fit = powerlaw.Fit(evals, xmin=xmin, xmax=xmax, verbose=False)
            
        alpha = fit.alpha 
        D = fit.D
        sigma = fit.sigma
        xmin = fit.xmin
        xmax = fit.xmax
        num_pl_spikes = len(evals[evals>=fit.xmin])

        if plot:
            fig2 = fit.plot_pdf(color='b', linewidth=2)
            fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig2)
            fit.plot_ccdf(color='r', linewidth=2, ax=fig2)
            fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig2)
        
            title = "Power law fit for {}\n".format(title) 
            title = title + r"$\alpha$={0:.3f}; ".format(alpha) + r"KS_distance={0:.3f}".format(D) + "\n"
            plt.title(title)
            plt.show()
    
            # plot eigenvalue histogram
            num_bins = 100  # np.min([100,len(evals)])
            plt.hist(evals, bins=num_bins, density=True)
            plt.title(r"ESD (Empirical Spectral Density) $\rho(\lambda)$" + "\nfor {} ".format(title))                  
            plt.axvline(x=fit.xmin, color='red', label='xmin')
            plt.legend()
            plt.show()

            # plot log eigenvalue histogram
            nonzero_evals = evals[evals > 0.0]
            plt.hist(np.log10(nonzero_evals), bins=100, density=True)
            plt.title(r"Log10 ESD (Empirical Spectral Density) $\rho(\lambda)$" + "\nfor {} ".format(title))                  
            plt.axvline(x=np.log10(fit.xmin), color='red')
            plt.axvline(x=np.log10(fit.xmax), color='orange', label='xmax')
            plt.legend()
            plt.show()
    
            # plot xmins vs D
            
            plt.plot(fit.xmins, fit.Ds, label=r'$D$')
            plt.axvline(x=fit.xmin, color='red', label='xmin')
            plt.plot(fit.xmins, fit.sigmas / fit.alphas, label=r'$\sigma /\alpha$', linestyle='--')
            plt.xlabel(r'$x_{min}$')
            plt.ylabel(r'$D,\sigma,\alpha$')
            plt.title("current xmin={:0.3}".format(fit.xmin))
            plt.legend()
            plt.show() 
                          
        return alpha, xmin, xmax, D, sigma, num_pl_spikes
    
    
    def get_ESD(self, model=None, layer=None, params=DEFAULT_PARAMS):
        """Get the ESD (empirical spectral density) for the layer, specified by id or name)"""
        
        model = self.model or model
        
        details = self.describe(model=model)
        layer_ids = details['layer_id'].to_numpy()
        layer_names = details['name'].to_numpy()
        
        if type(layer) is int and layer not in layer_ids:
            logger.error("Can not find layer id {} in valid layer_ids {}".format(layer, layer_ids))
            return []
        
        elif type(layer) is str and layer not in layer_names:
            logger.error("Can not find layer name {} in valid layer_names {}".format(layer, layer_names))
            return []
    
        logger.info("Getting ESD for layer {} ".format(layer))

        layer_iter = WWLayerIterator(model=model, filters=[layer], params=params)     
        details = pd.DataFrame(columns=['layer_id', 'name'])
           
        ww_layer = next(layer_iter)
        assert(not ww_layer.skipped) 
        assert(ww_layer.has_weights)
        
        self.apply_esd(ww_layer, params)
            
        esd = ww_layer.evals
        if esd is None or len(esd)==0:
            logger.warn("No eigenvalues found for {} {}".format(ww_layer.layer_id, ww_layer.name))
                
        else:
            logger.info("Found {} eiganvalues for {} {}".format(len(esd), ww_layer.layer_id, ww_layer.name))     
            
        return esd
    
    def apply_norm_metrics(self, ww_layer, params=DEFAULT_PARAMS):
        """Compute the norm metrics, as they depend on the eigenvalues"""

        layer_id = ww_layer.layer_id
        name = ww_layer.name or "" 
        evals = ww_layer.evals
        
        # TODO:  check normalization on all
        norm = np.sum(evals)
        log_norm = np.log10(norm)

        spectral_norm = np.max(evals)     
        log_spectral_norm = np.log10(spectral_norm)
        
        # TODO: check formula
        alpha = ww_layer.alpha
        alpha_weighted = alpha*log_spectral_norm
        log_alpha_norm = np.log10(np.sum( [ ev**alpha for ev in evals]))
        
        stable_rank = norm / spectral_norm
                    
        ww_layer.add_column(METRICS.NORM, norm)
        ww_layer.add_column(METRICS.LOG_NORM, log_norm)
        ww_layer.add_column(METRICS.SPECTRAL_NORM, spectral_norm)
        ww_layer.add_column(METRICS.LOG_SPECTRAL_NORM, log_spectral_norm)
        ww_layer.add_column(METRICS.ALPHA, alpha)
        ww_layer.add_column(METRICS.ALPHA_WEIGHTED, alpha_weighted)
        ww_layer.add_column(METRICS.LOG_ALPHA_NORM, log_alpha_norm)
        ww_layer.add_column(METRICS.STABLE_RANK, stable_rank)

        return ww_layer
    
    
    
    def apply_mp_fit(self, ww_layer, random=True, params=DEFAULT_PARAMS):
        """Perform MP fit on random or actual random eigenvalues
        N/A yet"""

        layer_id = ww_layer.layer_id
        name = ww_layer.name or ""
        layer_id_name = "{} {}".format(layer_id, name)
        
        if random:
            title = "Layer {} randomize W".format(layer_id_name)
            evals = ww_layer.rand_evals
        else:
            title = "Layer {} W".format(layer_id_name)
            evals = ww_layer.evals

        N, M = ww_layer.N, ww_layer.M
        

        num_spikes, sigma_mp, mp_softrank = self.mp_fit(evals, N, M, title, layer_id_name, params['plot'])
        
        if random:
            ww_layer.add_column('rand_num_spikes', num_spikes)
            ww_layer.add_column('rand_sigma_mp', sigma_mp)
            ww_layer.add_column('rand_mp_softrank', mp_softrank)
        else:
            ww_layer.add_column('num_spikes', num_spikes)
            ww_layer.add_column('sigma_mp', sigma_mp)
            ww_layer.add_column(METRICS.MP_SOFTRANK, mp_softrank)
            
        return 

    def mp_fit(self, evals, N, M, title, layer_id, plot):
        """Automatic MP fit to evals, compute numner of spikes and mp_softrank """
        
        Q = N/M
        lambda_max = np.max(evals)
        
        to_plot = evals.copy()
        
        bw = 0.1 
        s1, f1 = fit_density_with_range(to_plot, Q, bw = bw)
        sigma_mp = s1
        
        bulk_edge = (s1 * (1 + 1/np.sqrt(Q)))**2
        
        #TODO: add Tracy Widom (TW) range
        
        num_spikes = len(to_plot[to_plot > bulk_edge])
        ratio_numofSpikes  = num_spikes / (M - 1)
        
        mp_softrank = bulk_edge / lambda_max
        
        if Q == 1.0:
            fit_law = 'QC SSD'
            
            #Even if the quarter circle applies, still plot the MP_fit
            if plot:
                plot_density(to_plot, s1, Q, method = "MP")
                plt.legend([r'$\rho_{emp}(\lambda)$', 'MP fit'])
                plt.title("MP ESD, sigma auto-fit ")
                plt.show()
            
        else:
            fit_law = 'MP ESD'
#        

        plot_density_and_fit(model=None, eigenvalues=to_plot, layer=layer_id,
                              Q=Q, num_spikes=0, sigma=s1, verbose = False, plot=plot)
        
        if plot:
            title = fit_law+" "+title+"\n Q={:0.3} ".format(Q)
            title = title + r"$\sigma_{mp}=$"+"{:0.3} ".format(sigma_mp)
            title = title + r"$\mathcal{R}_{mp}=$"+"{:0.3} ".format(mp_softrank)
            title = title + r"$\#$ spikes={}".format(num_spikes)
    
            plt.title(title)
            plt.show()
            
        return num_spikes, sigma_mp, mp_softrank

        
        
        
        

   
        