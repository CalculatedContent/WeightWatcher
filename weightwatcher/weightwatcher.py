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
import sys, os, re, io
import logging

# Telly                                                                                        
import telly; import sys
telly.CONFIG["TELLY_OPTIN"] = False
telly.CONFIG["TELLY_CLIENT_ID"].append('801cb61e-2506-490f-8dd8-1a9d2d725ce2')
telly.CONFIG['TELLY_TIMER'] = True
telly.CONFIG['TELLY_TIMER_INTERVAL'] = 60*60*24

import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg

import matplotlib
import matplotlib.pyplot as plt
import powerlaw
 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import torch
import torch.nn as nn

import onnx
from onnx import numpy_helper


import sklearn
from sklearn.decomposition import TruncatedSVD

from copy import deepcopy

# remove warnings from powerlaw unless testing
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# for powerlaw warnings
from contextlib import redirect_stdout, redirect_stderr




#
# this is use to allow editing in Eclipse but also
# building on the commend line
# see: https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
#

from .RMT_Util import *
from .constants import *
from numpy import vectorize


WW_NAME = 'weightwatcher'
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(WW_NAME) 
logger.setLevel(logging.INFO)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)




def main():
    """
    Weight Watcher
    """
    print("WeightWatcher command line support coming later. https://calculationconsulting.com")


class ONNXLayer:
    """Helper class to support ONNX layers
    
    Turns out the op_type is option, so we have to 
    infers the layer_ type from the dimension of the weights 
        [a,b,c,d]  ->  CONV2D 
        [a,b]  ->  DENSE 
                
    """
    
    def __init__(self, model, inode, node):
        self.model = model
        self.node = node
        self.layer_id = inode
        self.name = node.name
        self.dims = node.dims
        self.the_type = LAYER_TYPE.UNKNOWN

        if len(self.dims) == 4:
            self.the_type = LAYER_TYPE.CONV2D
        elif len(self.dims) == 2:
            self.the_type = LAYER_TYPE.DENSE
        else:
            logger.debug("Unsupported ONNX Layer, dims = {}".format(self.dims))
            
    def get_weights(self):
        return numpy_helper.to_array(self.node) 
    
    def set_weights(self, idx, W):
        T = numpy_helper.from_array(W)
        self.model.graph.initializer[idx].CopyFrom(T)

        
        
class WWLayer:
    """WW wrapper layer to Keras and PyTorch Layer layer objects
       Uses pythong metaprogramming to add result columns for the final details dataframe"""
       
    def __init__(self, layer, layer_id=-1, name=None,
                 the_type=LAYER_TYPE.UNKNOWN, 
                 framework=FRAMEWORK.UNKNOWN, 
                 channels=CHANNELS.UNKNOWN,
                 skipped=False, make_weights=True, params=DEFAULT_PARAMS):
        self.layer = layer
        self.layer_id = layer_id  
        self.name = name
        self.skipped = skipped
        self.the_type = the_type
        self.framework = framework      
        self.channels = channels
        
        # get the LAYER_TYPE
        self.the_type = self.layer_type(self.layer)
        
        if self.name is None and hasattr(self.layer, 'name'):
            self.name = self.layer.name
        elif self.name is None:
            self.name = str(self.layer)
            self.name = re.sub(r'\(.*', '', self.name)

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
        
        # permute ids
        self.permute_ids = []
        
        # evals 
        self.evals = None
        self.rand_evals = None
        
        # details, set by metaprogramming in apply_xxx() methods
        self.columns = []
        
        # conv2d_fft
        # only applies to Conv2D layers
        # layer, this would be some kind of layer weight options
        self.params = params
        
        # original dimensions of the weight tensor for this layer
        self.weight_dims = None
        self.num_params = 0
        self.W_permuations = []
        
        # don't make if we set the weights externally
        if make_weights:
            self.make_weights()
            
        
    def add_column(self, name, value):
        """Add column to the details dataframe"""
        self.columns.append(name)
        self.__dict__[name] = value
        
    def get_value(self, name):
        """Get value of a column"""
        return self.__dict__[name]
    
    def get_column(self, name):
        """Get value of a column"""
        return self.__dict__[name]
    
    # does not work ?
    def has_column(self, name):
        """Get value of a column"""
        return name in self.__dict__
        
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
    
    
    def __repr__(self):
        return "WWLayer()"

    def __str__(self):
        return "WWLayer {}  {} {} {}  skipped {}".format(self.layer_id, self.name,
                                                       self.framework.name, self.the_type.name, self.skipped)
        
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

        elif  'norm' in str(type(layer)).lower():
            the_type = LAYER_TYPE.NORM

        # ONNX
        elif isinstance(layer,ONNXLayer):
            the_type = layer.the_type

        # allow user to specify model type with file mapping
        
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
        """extract the original weights (as a tensor) for the layer, and biases for the layer, if present
        """
        
        has_weights, has_biases = False, False
        weights, biases = None, None
    
        if self.framework == FRAMEWORK.PYTORCH:
            if hasattr(self.layer, 'weight'):
                w = [np.array(self.layer.weight.data.clone().cpu())]
                if self.the_type==LAYER_TYPE.CONV2D:
                    weights = w[0]
                    biases = None
                    has_weights = True
                elif self.the_type==LAYER_TYPE.CONV1D:
                    weights = w[0]
                    biases = None
                    has_weights = True
                elif self.the_type==LAYER_TYPE.EMBEDDING:
                    weights = w[0]
                    biases = None
                    has_weights = True
                elif self.the_type==LAYER_TYPE.DENSE:
                    weights = w[0]
                    #biases = w[1]
                    has_weights = True
                    #has_biases = True
                elif self.the_type not in [LAYER_TYPE.NORM]: 
                    logger.info("pytorch layer: {}  type {} not found ".format(str(self.layer),str(self.the_type)))
                else:
                    pass

                
        elif self.framework == FRAMEWORK.KERAS:
            w = self.layer.get_weights()
            if self.the_type==LAYER_TYPE.CONV2D:
                weights = w[0]
                biases = None
                has_weights = True
            elif self.the_type==LAYER_TYPE.CONV1D:
                weights = w[0]
                biases = None
                has_weights = True
            elif self.the_type==LAYER_TYPE.EMBEDDING:
                weights = w[0]
                biases = None
                has_weights = True
            elif self.the_type==LAYER_TYPE.DENSE:
                weights = w[0]
                biases = w[1]
                has_weights = True
                has_biases = True
            else: 
                logger.info("keras layer: {} {}  type {} not found ".format(self.layer.name,str(self.layer),str(self.the_type)))
                

        elif self.framework == FRAMEWORK.ONNX:      
            onnx_layer = self.layer
            weights = onnx_layer.get_weights()
            has_weights = True
        
        return has_weights, weights, has_biases, biases  
      
    def set_weight_matrices(self, weights):#, conv2d_fft=False, conv2d_norm=True):
        """extract the weight matrices from the framework layer weights (tensors)
        sets the weights and detailed properties on the ww (wrapper) layer 
    
        conv2d_fft not supported yet """
   
        if not self.has_weights:
            logger.info("Layer {} {} has no weights".format(self.layer_id, self.name))
            return 
        
        the_type = self.the_type
        conv2d_fft = self.params[CONV2D_FFT]
        
        N, M, n_comp, rf = 0, 0, 0, None
        Wmats = []
        
        # this may change if we treat Conv1D differently layer
        if (the_type == LAYER_TYPE.DENSE or the_type == LAYER_TYPE.CONV1D or the_type==LAYER_TYPE.EMBEDDING):
            Wmats = [self.weights]
            N, M = np.max(Wmats[0].shape), np.min(Wmats[0].shape)
            n_comp = M
            rf = 1
            
        # this is very slow with describe 
        elif the_type == LAYER_TYPE.CONV2D:
            if not conv2d_fft:
                Wmats, N, M, rf = self.conv2D_Wmats(weights, self.channels)
                n_comp = M*rf # TODO: bug fixed, check valid
            else:
                Wmats, N, M, n_comp = self.get_conv2D_fft(weights)
            
        elif the_type == LAYER_TYPE.NORM:
            #logger.info("Layer id {}  Layer norm has no matrices".format(self.layer_id))
            pass
        
        else:
            logger.info("Layer id {}  unknown type {} layer  {}".format(self.layer_id, the_type, type(self.layer)))
    
        self.N = N
        self.M = M
        self.rf = rf
        self.Wmats = Wmats
        self.num_components = n_comp
        
        self.weight_dims = self.weights.shape
        self.num_params = np.prod(self.weight_dims)
        
        return 
        
        
    def get_conv2D_fft(self, W, n=32):
        """Compute FFT of Conv2D CHANNELS, to apply SVD later"""
        
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

    
    def channel_str(self, channel):
        if channel==CHANNELS.FIRST:
            return "FIRST"
        elif channel==CHANNELS.LAST:
            return "LAST"
        else:
            return "UNKNOWN"
        
    def conv2D_Wmats(self, Wtensor, channels=CHANNELS.UNKNOWN):
        """Extract W slices from a 4 layer_id conv2D tensor of shape: (N,M,i,j) or (M,N,i,j).  
        Return ij (N x M) matrices, with receptive field size (rf) and channels flag (first or last)"""
        
        logger.debug("conv2D_Wmats")
        
        # TODO:  detect or use CHANNELS
        # if channels specified ...
    
        Wmats = []
        s = Wtensor.shape
        N, M, imax, jmax = s[0], s[1], s[2], s[3]
        
        if N + M >= imax + jmax:
            detected_channels= CHANNELS.LAST
        else:
            detected_channels= CHANNELS.FIRST
            

        if channels == CHANNELS.UNKNOWN :
            logger.debug("channels UNKNOWN, detected {}".format(self.channel_str(detected_channels)))
            channels= detected_channels

        if detected_channels == channels:
            if channels == CHANNELS.LAST:
                logger.debug("channels Last tensor shape: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))                
                for i in range(imax):
                    for j in range(jmax):
                        W = Wtensor[:, :, i, j]
                        if W.shape[0] < W.shape[1]:
                            N, M = M, N
                            W = W.T
                        Wmats.append(W)
                        
            else: #channels == CHANNELS.FIRST  # i, j, M, N
                M, N, imax, jmax = imax, jmax, N, M
                # check this       
                logger.debug("channels First shape: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))                
                for i in range(imax):
                    for j in range(jmax):
                        W = Wtensor[i, j, :, :]
                        if W.shape[1] < W.shape[0]:
                            N, M = M, N
                            W = W.T
                        Wmats.append(W)
                            
        elif detected_channels != channels:
            logger.warning("warning, expected channels {},  detected channels {}".format(self.channel_str(channels),self.channel_str(detected_channels)))
            # flip how we extract the WMats
            # reverse of above extraction
            if detected_channels == CHANNELS.LAST:
                logger.debug("Flipping LAST to FIRST Channel, {}x{} ()x{}".format(N, M, imax, jmax))   
                for i in range(N):
                    for j in range(M):
                        W = Wtensor[i, j,:,:]
                        if imax < jmax:
                            W = W.T
                        Wmats.append(W)
                        
            else: #detected_channels == CHANNELS.FIRST:
                N, M, imax, jmax = imax, jmax, N, M   
                logger.debug("Flipping FIRST to LAST Channel, {}x{} ()x{}".format(N, M, imax, jmax))                
                # check this       
                for i in range(N):
                    for j in range(M):
                        W = Wtensor[:, :, i, j]
                        if imax < jmax:
                            W = W.T
                        Wmats.append(W)
            # final flip            
            N, M, imax, jmax = imax, jmax, N, M   
           
                
        rf = imax * jmax  # receptive field size             
        logger.debug("get_conv2D_Wmats N={} M={} rf= {} channels= {}".format(N, M, rf, channels))
    
        return Wmats, N, M, rf
    
    


    def permute_Wmats(self):
        """randomly permute the weights in a way they can unpermuted deterministically"""
        
        self.W_permuted_ids = []
        p_Wmats = []
        for W in self.Wmats:
            p_W, p_ids = permute_matrix(W)
            p_Wmats.append(p_W)
            self.W_permuted_ids.append(p_ids)
            
        return p_Wmats
            
    def unpermute_Wmats(self, Wmats):
        """unpremute the previously permuted, randomized weights"""
        
        unp_Wmats = []
        for W, p_ids in zip(Wmats, self.W_permuted_ids):       
            unp_W = unpermute_matrix(W, p_ids)
            unp_Wmats.append(unp_W)
            
        self.W_permuted_ids = []
            
        return unp_Wmats
    
    def flatten_weights(self):
        """Transform the original weights tensor into a vector"""
        return self.weights.reshape(self.num_params)
    
    def unflatten_weights(self, vec):
        """unflatten the vector back to the original tensor weights"""
        return vec.reshape(self.weight_dims)
    
    def reset_weights(self, W):
        """reset the layer framework weight tensor"""
        logger.fatal("not implemented yet")
        return
    

    
class ModelIterator:
    """Iterator that loops over ww wrapper layers, with original matrices (tensors) and biases (optional) available."""

    def __init__(self, model, params=DEFAULT_PARAMS):
        
        self.params = params
        self.k = params[START_IDS] # 0 | 1
        
        logger.debug("FRAMEWORKS: KERAS = {}  PYTORCH = {} ONNX = {} UNKNOWN = {} ".format(FRAMEWORK.KERAS, FRAMEWORK.PYTORCH, FRAMEWORK.ONNX, FRAMEWORK.UNKNOWN))
        logger.debug("FIRST = {}  LAST = {} UNKNOWN = {} ".format(CHANNELS.FIRST, CHANNELS.LAST, CHANNELS.UNKNOWN))

        self.model = model
        self.framework = self.set_framework()
        self.channels  = self.set_channels(params.get(CHANNELS_STR))
        
        logger.debug("MODEL ITERATOR, framework = {}, channels= {} ".format(self.framework, self.channels))

        self.model_iter = self.model_iter_(model) 
        self.layer_iter = self.make_layer_iter_()            
        
        
    def set_framework(self):
        """infer the framework """
        
        framework = FRAMEWORK.UNKNOWN
        if hasattr(self.model, LAYERS):
            framework = FRAMEWORK.KERAS

        elif hasattr(self.model, 'modules'):
            framework = FRAMEWORK.PYTORCH

        elif isinstance(self.model, onnx.onnx_ml_pb2.ModelProto):  
            framework = FRAMEWORK.ONNX
        return framework
    
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
        
        if self.framework == FRAMEWORK.KERAS:
            def layer_iter_():

                def traverse_(layer):
                    "not recursive, just iterate over all submodules if present"
                    if not hasattr(layer, 'submodules') or len(layer.submodules)==0:
                        yield layer
                    else:                        
                        for sublayer in layer.submodules:
                            yield sublayer
                    
                for layer in model.layers:
                    yield from traverse_(layer)

            layer_iter = layer_iter_()


        elif self.framework == FRAMEWORK.PYTORCH:
            def layer_iter_():
                for layer in model.modules():
                        yield layer                        
            layer_iter = layer_iter_()    
            

        elif self.framework == FRAMEWORK.ONNX:
            def layer_iter_():
                for inode, node in enumerate(model.graph.initializer):
                    yield ONNXLayer(model, inode, node)                        
            layer_iter = layer_iter_()    
            
        else:
            layer_iter = None
            
        return layer_iter
                      
    def make_layer_iter_(self):
        """The layer iterator for this class / instance.
         Override this method to change the type of iterator used by the child class"""
        return self.model_iter
    
    def set_channels(self, channels=None):
        """Set the channels flag for the framework, with the ability to override"""
                
        the_channel = CHANNELS.UNKNOWN
        if channels is None:
            if self.framework == FRAMEWORK.KERAS:
                the_channel = CHANNELS.FIRST
                
            elif self.framework == FRAMEWORK.PYTORCH:
                the_channel = CHANNELS.LAST
                
            elif self.framework == FRAMEWORK.ONNX:
                the_channel = CHANNELS.LAST
        elif channels(channels, str):
            if channels.lower()=='first':
                the_channel=CHANNELS.FIRST
                
            elif channels.lower()=='last':
                the_channel=CHANNELS.LAST
                
        return the_channel


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
                logger.warning("unknown filter type {} detected and ignored".format(tf))
                
    def apply_filters(self, ww_layer):
        """Apply filters.  Set skipped False  if filter is applied to this layer, keeping the layer (or no filters, meaning all layers kept)"""
        ww_layer.skipped = False
          
        if self.filter_types is not None and len(self.filter_types) > 0:
            if ww_layer.the_type in self.filter_types:
                logger.debug("keeping layer {} {} with type {} ".format(ww_layer.layer_id, ww_layer.name , str(ww_layer.the_type)))
                ww_layer.skipped = False
            else:
                logger.debug("skipping layer {} {} with type {} ".format(ww_layer.layer_id, ww_layer.name , str(ww_layer.the_type)))
                ww_layer.skipped = True

        
        if self.filter_ids is not None and len(self.filter_ids) > 0:
            # keep positive layer ids
            if np.min(self.filter_ids) > 0:
                if ww_layer.layer_id in self.filter_ids:
                    logger.debug("keeping layer {} {} by id".format(ww_layer.layer_id, ww_layer.name))
                    ww_layer.skipped = False
                else:
                    logger.debug("skipping layer {} {} by id".format(ww_layer.layer_id, ww_layer.name))
                    ww_layer.skipped = True
            # or remove negative layer ids
            elif np.min(self.filter_ids) < 0:
                if -(ww_layer.layer_id) in self.filter_ids:
                    logger.debug("skipping layer {} {} by id".format(ww_layer.layer_id, ww_layer.name))
                    ww_layer.skipped = True
                else:
                    logger.debug("keeping layer {} {} by id".format(ww_layer.layer_id, ww_layer.name))
                    ww_layer.skipped = False


                
        if self.filter_names is not None and len(self.filter_names) > 0:
            if ww_layer.name in self.filter_names:
                logger.debug("keeping layer {} {} by name ".format(ww_layer.layer_id, ww_layer.name))
                ww_layer.skipped = False
            else:
                logger.debug("skipping layer {} {} by name ".format(ww_layer.layer_id, ww_layer.name))
                ww_layer.skipped = True
     
        return ww_layer.skipped
    
    def ww_layer_iter_(self):
        """Create a generator for iterating over ww_layers, created lazily """
        for curr_layer in self.model_iter:
            curr_id, self.k = self.k, self.k + 1
            
            ww_layer = WWLayer(curr_layer, layer_id=curr_id, 
                               framework=self.framework, 
                               channels=self.channels,
                               params=self.params)
            
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

        ww2x = self.params.get(WW2X)
        
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
        
        
        elif ww2x and min_evals and M  <  min_evals:
            logger.debug("layer not supported: Layer {} {}: num_evals {} <  min_evals {}".format(layer_id, name, M, min_evals))
            return False
                  
        elif ww2x and max_evals and M  >  max_evals:
            logger.debug("layer not supported: Layer {} {}: num_evals {} > max_evals {}".format(layer_id, name, N, max_evals))
            return False

        elif (not ww2x) and min_evals and M * rf < min_evals:
            logger.debug("layer not supported: Layer {} {}: num_evals {} <  min_evals {}".format(layer_id, name, M * rf, min_evals))
            return False
                  
        elif (not ww2x) and max_evals and M * rf > max_evals:
            logger.debug("layer not supported: Layer {} {}: num_evals {} > max_evals {}".format(layer_id, name, N * rf, max_evals))
            return False
        
        elif the_type in [LAYER_TYPE.DENSE, LAYER_TYPE.CONV1D, LAYER_TYPE.CONV2D, LAYER_TYPE.EMBEDDING]:
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
    


class WWIntraLayerIterator(WW2xSliceIterator):
    """Iterator variant that iterates over N-1 layer pairs, forms ESD for cross correlations
    
    Note:  apply_esd computes eigenvalues
           for intra-layer fits, we need the singular values of X, not the eigenvalues
           so, for coinsistancy with other methods, we need to mix the notation
    
    """
    from copy import deepcopy
    
    prev_layer = None

    def ww_intralayer_iter_(self): 
               
        # TODO: detect the layer ordering and flip accordingly
        # for  all layers the same way
        def align_mats(W0, W1):
            """align the mats so that one can take X = np.dot(W0,W1)
            
            i.e:  W0.shape[1]==W1.shape[0] along the shared dinension  
            """
            
            logger.info("aligning {} {}".format(W0.shape, W1.shape))      
            N0, M0 = np.max(W0.shape),  np.min(W0.shape)
            N1, M1 = np.max(W1.shape),  np.min(W1.shape)
            
            shapes = [W0.shape[0], W0.shape[1], W1.shape[0], W1.shape[1]]
            N, M = np.max(shapes), np.min(shapes)
            
            # do these arrays share a dimension ?
            shared_dim = None
            for x in np.unique(shapes):
                shapes.remove(x)
            if len(shapes) > 0:
                shared_dim = np.max(shapes)
                    
                logger.debug("found shared dim = {}".format(shared_dim))
                    
                if not shared_dim:
                    logger.warning("Can not align W0={} with W1={}".format(W0.shape, W1.shape))
                    return W0, W1
                    
                if W0.shape[1] != shared_dim:
                    W0 = np.transpose(W0)
                        
                if W1.shape[0] != shared_dim:
                    W1 = np.transpose(W1)
                            
            return W0, W1
                        
   
        ## Need to look at all W, currently just doing 1
        for ww_layer in self.ww_layer_iter_():
            if self.prev_layer is None:
                self.prev_layer = deepcopy(ww_layer)
            else:
                name = "{} X {} ".format(self.prev_layer.layer_id, ww_layer.layer_id)
                logger.info("Analyzing {}  ".format(name))

                W0 = self.prev_layer.Wmats[0]                                                                  
                W1 = ww_layer.Wmats[0]
                W0, W1 = align_mats(W0, W1)
                
                self.prev_layer = deepcopy(ww_layer)
                ww_intralayer = deepcopy(ww_layer)
                ww_intralayer.name = name
                
                # NEED TO LOOK AT ALL LAYERS
                ww_intralayer.count = 1
                
                sparsify=self.params[SPARSIFY]
                
                if W0.shape[1]!=W1.shape[0]:
                    logger.info(" {} not compatible, skipping".format(name))
                else:            
                    norm12 = np.linalg.norm(W0)*np.linalg.norm(W1)  # probably wrong 
                    if sparsify:
                        logger.info("sparsifying overlaps")
                        M = np.min(W1.shape[0])
                        S = np.random.randint(2, size=M*M)
                        S = S.reshape((M,M))
                        norm12 = norm12+np.sqrt(np.linalg.norm(S)) # probably wrong 
                        X = np.dot(np.dot(W0,S),W1)/(norm12)
                    else:
                        logger.info("direct overlaps")
                        X = np.dot(W0,W1)/(norm12)
                    ww_intralayer.Wmats = [X]
                    ww_intralayer.N = np.max(X.shape)
                    ww_intralayer.M = np.min(X.shape)
                    
                    ww_intralayer.add_column("Xflag", True)
                  
                    yield ww_intralayer

                
    def make_layer_iter_(self):
        return self.ww_intralayer_iter_()
    
    
class WWStackedLayerIterator(WWLayerIterator):
    """Iterator variant that stcaks all weight matrices into a single WWLayer
    
    Notes: 
    - Only supports ww2x=False 
    
    - The layer can be analyzed, but does not yet support SVDSmoothing, etc 
    
    - Each layer matrix is normalized by the Frobenius norm  W=W/||W||_F

    - Each layer matric is padded with zeros to the right,  i.e
    
        [1, 2, 0, 0, 0, 0]
        [3, 4, 0, 0, 0, 0]
        [5, 6, 7, 8, 0, 0]
        [9, 9, 9, 9, 0, 0]
        ...
    
    """

    def ww_stacked_iter_(self):
        
        # find the maximum dimensions so we can pad the matrices
        ww_stacked_layer = None
        Wmats = []
        for ww_layer in self.ww_layer_iter_():
            
            # Here, I just lazizy copy an older layer
            # really, we should creat the WWLayer using the init() constructor
            if ww_stacked_layer is None:
                ww_stacked_layer =  deepcopy(ww_layer)
                ww_stacked_layer.the_type =  LAYER_TYPE.STACKED
                ww_stacked_layer.layer_id = 0  
                ww_stacked_layer.name = "Stacked Layer"
                
            # first attempt
            #Wmats.extend(ww_layer.Wmats)
            
            # second attempt
            #  stack all the conv2d matrices horizontally first
            if len(ww_layer.Wmats)>1:
                W = np.hstack(ww_layer.Wmats)
            else:
                W = ww_layer.Wmats[0]   
                
            #N, M =  W.shape[0],  W.shape[1]
            #if N < M:
            #    W = W.T            
            Wmats.append(W)
             
        #  Layer Matrices  are padded with zeros 
        #   i.e: [1,2,3,4, 0,0,0,0] so to the same width
        # 
        
        Ms = [int(W.shape[1]) for W in Wmats]
        
        Mmax, Mmin = np.max(Ms), np.min(Ms)
                        
        Wmats_padded = []
        for W in Wmats:             
            Height, Width = W.shape[0], W.shape[1]
#            fixed above
#            if Height > Width:
#                W = W.T
            #Height, Width = W.shape[0], W.shape[1]
               
            #W = W/np.linalg.norm(W)
            W = (W - np.median(W))/sp.stats.median_abs_deviation(W)
            W = np.pad(W, ((0, 0), (0, Mmax-Width)) ) 
            Wmats_padded.append(W)
                
        W_stacked = np.vstack(Wmats_padded)
        N, M = W_stacked.shape[0],  W_stacked.shape[1]
        if N < M:
            W_stacked = W_stacked.T
            N, M = W_stacked.shape[0],  W_stacked.shape[1]
                    
        ww_stacked_layer.Wmats = [W_stacked]

        ww_stacked_layer.N = N
        ww_stacked_layer.M = M
        ww_stacked_layer.rf = 1
        
        
        # Thi needs to be reset and used properly , eventually
        # the effective M, used for Q, will be much smaller 
        # if there are a huge number of zero eigenvalues 
        self.num_components = M  # default for full SVD, not used yet
        
        yield ww_stacked_layer
                
    def make_layer_iter_(self):
        return self.ww_stacked_iter_()
    
    

    
class WeightWatcher(object):

    def __init__(self, model=None, log_level=None):
        if log_level:
            logger.setLevel(log_level)

        self.model = self.load_model(model)
        self.details = None
        logger.info(self.banner())


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
    
    @telly.count_decorator
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
                        
                    # issue 137
                    #details = details.append(data, ignore_index=True)
                    data_df = pd.DataFrame.from_records(data , index=[0])
                    details = pd.append([details, data_df])
        except:
            logger.error("Sorry, problem comparing models")
            raise Exception("Sorry, problem comparing models")
        
        details.set_index('layer_id', inplace=True)
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
        normalize = params[NORMALIZE]
        glorot_fix = params[GLOROT_FIX]
        conv2d_norm = params[CONV2D_NORM]  # True
        
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
            sv = sp.linalg.svd(W, compute_uv=False)
            sv = sv.flatten()
            sv = np.sort(sv)[-n_comp:]
            # TODO:  move to PL fit for robust estimator
            # if len(sv) > max_evals:
            #    #logger.info("chosing {} singular values from {} ".format(max_evals, len(sv)))
            #    sv = np.random.choice(sv, size=max_evals)
    
            # sv = svd.singular_values_
            if params[INTRA]:
                evals = sv
                sv  = np.sqrt(sv)
            else:
                evals = sv * sv
            
            #if normalize:
            #    evals = evals / N
    
            all_evals.extend(evals)
    
            max_sv = np.max([max_sv, np.max(sv)])
            rank_loss = rank_loss + calc_rank_loss(sv, N)      
    
        return np.sort(np.array(all_evals)), max_sv, rank_loss
            
            
    def apply_normalize_Wmats(self, ww_layer, params=DEFAULT_PARAMS):
        """Normalize the W matrix or Wmats """

        normalize = params[NORMALIZE]
        glorot_fix = params[GLOROT_FIX]
        conv2d_norm = params[CONV2D_NORM]
        
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

        # TODO: set np linalg norm, <AVG over layers>
        # change from NORM that is computed...
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
        
        if params[TOLERANCE]:
            tolerance = params[TOLERANCE]
        else:
            tolerance = WEAK_RANK_LOSS_TOLERANCE
        weak_rank_loss = len(evals[evals<tolerance])
     
        ww_layer.evals = evals
        ww_layer.add_column("has_esd", True)
        ww_layer.add_column("num_evals", len(evals))
        ww_layer.add_column("sv_max", sv_max)
        ww_layer.add_column("rank_loss", rank_loss)
        ww_layer.add_column("weak_rank_loss", weak_rank_loss)
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

        # measure distance between random and non-random esd
        #  https://medium.com/@sourcedexter/how-to-find-the-similarity-between-two-probability-distributions-using-python-a7546e90a08d
        evals = ww_layer.evals
        if evals is not None and len(evals>0):
            rand_evals = self.random_eigenvalues(Wmats, n_comp, 1 , params)

            value = jensen_shannon_distance(evals, rand_evals)
            ww_layer.add_column("rand_distance", value)

            value = np.max(evals)/np.max(rand_evals)
            ww_layer.add_column("ww_softrank", value)

            value = np.max(evals)-np.max(rand_evals)
            ww_layer.add_column("ww_maxdist", value)

        if params[PLOT]:
            self.plot_random_esd(ww_layer, params)
            
        return ww_layer
    
    
    def apply_permute_W(self, ww_layer, params=DEFAULT_PARAMS):
        """Randomize the layer weight matrices by using a deterministic permutation
        This will replace the WMats ; they can be recovered by apply_unpermute_W()
         """
        
        layer_id = ww_layer.layer_id
        name = ww_layer.name
        
        logger.debug("apply permute W  on Layer {} {} ".format(layer_id, name))                        
        logger.debug("params {} ".format(params))
    
        Wmats, permute_ids = [], []
        for W in ww_layer.Wmats:
            W, p_ids = permute_matrix(W)
            Wmats.append(W)
            permute_ids.append(p_ids)
                           
        ww_layer.Wmats = Wmats
        ww_layer.permute_ids = permute_ids
        
        return ww_layer
    
    
      
    def apply_unpermute_W(self, ww_layer, params=DEFAULT_PARAMS):
        """Unpermute the layer weight matrices after the deterministic permutation
        This will replace the WMats ; only works if applied after  apply_permute_W()
         """
        
        layer_id = ww_layer.layer_id
        name = ww_layer.name
        
        logger.debug("apply unpermute W  on Layer {} {} ".format(layer_id, name))                        
        logger.debug("params {} ".format(params))
    
        Wmats = []
        for W, p_ids in zip(ww_layer.Wmats, ww_layer.permute_ids):
            W = unpermute_matrix(W, p_ids)
            Wmats.append(W)
                           
        ww_layer.Wmats = Wmats
        ww_layer.permute_ids = []
        
        return ww_layer
    
    
        # Not used yet
    def apply_detX(self, ww_layer, params=DEFAULT_PARAMS):
        """Compute the detX constraint, and optionally plot """
                    
        plot = params[PLOT]           
        savefig = params[SAVEFIG]
        savedir = params[SAVEDIR]

        evals = ww_layer.evals        
        evals = rescale_eigenvalues(evals)
        detX_num, detX_idx = detX_constraint(evals, rescale=False)
        detX_val = evals[detX_idx]

        ww_layer.add_column('detX_num', detX_num)  
        ww_layer.add_column('detX_val', detX_val)  

        if plot:
            name = ww_layer.name
            # fix rescaling to plot xmin

            layer_id = ww_layer.layer_id  # where is the layer_id
            plt.title(f"DetX constraint for {name}")
            plt.xlabel("log10 eigenvalues (norm scaled)")
            plt.hist(np.log10(evals), bins=100)
            plt.axvline(np.log10(detX_val), color='purple', label=r"detX$=1$")
            
            if ww_layer.xmin:
                xmin = ww_layer.xmin *  np.max(evals)/ww_layer.xmax
                plt.axvline(np.log10(xmin), color='red', label=r"PL $\lambda_{min}$")
                
            plt.legend()
            if savefig:
                save_fig(plt, "detX", layer_id, savedir)
            plt.show(); plt.clf()
            
        return ww_layer
    
    
    # Not used yet
    def apply_plot_esd(self, ww_layer, params=DEFAULT_PARAMS):
        """Plot the ESD on regular and log scale.  Only used when powerlaw fit not called"""
                    
        evals = ww_layer.evals
        name = ww_layer.name
        
        plt.title(name)
        plt.hist(evals, bins=100)
        plt.show(); plt.clf()
        
        plt.title(name)
        plt.hist(np.log10(evals), bins=100)
        plt.show(); plt.clf()
            
        return ww_layer
    
    
 
    def apply_fit_powerlaw(self, ww_layer, params=DEFAULT_PARAMS):
        """Plot the ESD on regular and log scale.  Only used when powerlaw fit not called"""
                    
        evals = ww_layer.evals
        layer_id = ww_layer.layer_id
        name = ww_layer.name
        title = "{} {}".format(layer_id, name)

        xmin = None  # TODO: allow other xmin settings
        xmax = np.max(evals)
        plot = params[PLOT]
        sample = False  # TODO:  decide if we want sampling for large evals       
        sample_size = None

        savefig = params[SAVEFIG]
        savedir = params[SAVEDIR]

        ff =  params[FIX_FINGERS]
        layer_name = "Layer {}".format(layer_id)
        
        fit_type =  params[FIT]

        alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, best_fit, status = self.fit_powerlaw(evals, xmin=xmin, xmax=xmax, plot=plot, layer_name=layer_name, layer_id=layer_id, sample=sample, sample_size=sample_size, savedir=savedir, savefig=savefig, fix_fingers=ff, fit_type=fit_type)

        ww_layer.add_column('alpha', alpha)
        ww_layer.add_column('xmin', xmin)
        ww_layer.add_column('xmax', xmax)
        ww_layer.add_column('D', D)
        ww_layer.add_column('sigma', sigma)
        ww_layer.add_column('num_pl_spikes', num_pl_spikes)
        ww_layer.add_column('best_fit', best_fit) 
        ww_layer.add_column('Lambda', Lambda) #-1 for PL, 
   
        ww_layer.add_column('warning', status)

        return ww_layer


    def make_layer_iterator(self, model=None, layers=[], params=DEFAULT_PARAMS):
        """Constructor for the Layer Iterator; See analyze(...)
        
         """
         
        # this doesn't seem to work
        if model is None:
            model = self.model
            
        logger.info("params {}".format(params))
        if not self.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)

        #stacked = params['stacked']
        intra = params[INTRA]
        ww2x = params[WW2X]
        stacked = params[STACKED]
        
        layer_iterator = None
        if stacked:
            logger.info("Using Stacked Iterator (experimental)")
            layer_iterator = WWStackedLayerIterator(model, filters=layers, params=params)    
        elif intra:
            logger.info("using Intra layer Analysis (experimental)")
            layer_iterator = WWIntraLayerIterator(model, filters=layers, params=params)     
        elif ww2x:
            logger.info("Using weightwatcher 0.2x style layer and slice iterator")
            layer_iterator = WW2xSliceIterator(model, filters=layers, params=params)     
        else:
            layer_iterator = WWLayerIterator(model, filters=layers, params=params)     
    
        return layer_iterator
    
    
        
    # test with https://github.com/osmr/imgclsmob/blob/master/README.md
    @telly.count_decorator
    def analyze(self, model=None, layers=[], 
                min_evals=DEFAULT_MIN_EVALS, max_evals=DEFAULT_MAX_EVALS,
                min_size=None, max_size=None, 
                glorot_fix=False,
                plot=False, randomize=False,  
                savefig=DEF_SAVE_DIR,
                mp_fit=False, conv2d_fft=False, conv2d_norm=True,  ww2x=False,
                deltas=False, intra=False, vectors=False, channels=None, 
                stacked=False, fix_fingers=False, fit=PL, sparsify=True, 
                detX=False, 
                tolerance=WEAK_RANK_LOSS_TOLERANCE,
                start_ids=0):
        """
        Analyze the weight matrices of a model.

        Parameters
        ----------
        
        layers:
            List of layer ids. If empty, analyze all layers (default)
            If layer ids < 0, then skip the layers specified
            All layer ids must be > 0 or < 0
            
        min_evals:  int, default=50
            Minimum number of evals (M*rf) 
            
        max_evals:  int, default=10000
            Maximum number of evals (N*rf) (0 = no limit)
            
        #removed
        normalize:  bool, default: True
            Normalize the X matrix. Usually True for Keras, False for PyTorch.
            Ignored if glorot_norm is set
            
        glorot_fix:  bool, default: False
            Adjust the norm for the Glorot Normalization.
              
    
        mp_fit:  bool, default: False
            Compute the best Marchenko-Pastur fit of each weight matrix ESD
            For square matrices, also applies the Quarter-Circle (QC) law
            
        randomize:  bool, default: False
            Randomizes the W matrices, plots the ESD and fits to the MP distribution
            Attempts to find Correlation Traps by computing the number of spikes for the randomized ESD 
            
        conv2d_fft:  N/A yet
            For Conv2D layers, use FFT method.  Otherwise, extract and combine the weight matrices for each receptive field
            Note:  for conf2d_fft, the ESD is automatically subsampled to max_evals eigenvalues max  N/A yet
            Can not uses with ww2x
            
        ww2x:  bool, default: False
            Use weightwatcher version 0.2x style iterator, which slices up Conv2D layers in N=rf matrices
            
        savefig:  string,  default: 
            Save the figures generated in png files.  Default: save to ww-img
            If set to a folder name, creates and saves the imafes to this folder (i.e. savefig="images")
            
        rescale:  #deprecated, always True
            Rescale the ESDs when computing the MP fits (experimental should always be True
            N/A yet: rescales the plots back to the original scale
            
        deltaEs:  bool, default: False
            Compute and plot the deltas of the eigenvalues; only works if plot=True. 
            Plots both as a sequence of deltaEs and a histogram (level statistics
            
        channels: None | 'first' | 'last'
            re/set the channels from the default for the framework
            
        vectors:  bool, default: False
            Compute the eigenvectors and plots various metrics, including the vector entropy and localization statistics, 
            both as a sequence (elbow plots) and as histograms
            Warning:  this takes more memory and some time
            
        stacked:   bool, default: False  (experimental)
            Stack all the weight matrices into a single Layer, and analyze
            Can be very slow.
            
        fix_fingers:  bool, default: False 
            Attempts to fix the fingers / finite-size-effects which sometimes occurs layers with spuriously large alphas
            Can be very slow.
            
        fit:  string, default: 'PL'
            PL (power_law) or TPL (truncated_power_law), E_TPL (extended TPL)
            In principle, we could return both, alpha and the TPL alpha, Lambda
            
        intra:  bool, default: False 
            Analyze IntraLayer Correlations
            Experimental option
            
        sparsify:  bool, default: True 
            only relevant for intra
            applies sparsify transformation to simulate ReLu(s) between layers
            maybe we don't want for certain transformer layers ?
            
        detX:  bool, default: False 
            compute the Trace Log Norm / DetX=1 constraint, and plot if plot True
            
        tolerance: float, default 0.000001
            sets 'weak_rank_loss' = number of  eigenvalues <= tolerance
          
        params:  N/A yet
            a dictionary of default parameters, which can be set but will be over-written by 

        start_ids:  0 | 1
           Start layer id counter at 0 or 1
        """

        model = model or self.model   
        
        if min_size or max_size:
            logger.warning("min_size and max_size options changed to min_evals, max_evals, ignored for now")     
        
        # I need to figure this out
        # can not specify params on input yet
        # maybe just have a different analyze() that only uses this 
        
        params=DEFAULT_PARAMS
        params[MIN_EVALS] = min_evals 
        params[MAX_EVALS] = max_evals
        params[PLOT] = plot
        params[RANDOMIZE] = randomize
        params[MP_FIT] = mp_fit
        #params[NORMALIZE] = normalize   #removed 0.6.5
        params[GLOROT_FIT] = glorot_fix
        params[CONV2D_NORM] = conv2d_norm
        params[CONV2D_FFT] = conv2d_fft
        params[WW2X] = ww2x   
        params[DELTA_ES] = deltas 
        params[INTRA] = intra 
        params[CHANNELS_STR] = channels
        params[LAYERS] = layers
        params[VECTORS] = vectors
        params[STACKED] = stacked
        params[FIX_FINGERS] = fix_fingers
        params[FIT] = fit
        params[SPARSIFY] = sparsify
        params[DETX] = detX
        params[TOLERANCE] = tolerance
        params[START_IDS] = start_ids


        params[SAVEFIG] = savefig
        #params[SAVEDIR] = savedir

            
        logger.debug("params {}".format(params))
        if not self.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)
        
        layer_iterator = self.make_layer_iterator(model=model, layers=layers, params=params)     
        
        details = pd.DataFrame(columns=['layer_id', 'name'])
        
        for ww_layer in layer_iterator:
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.debug("LAYER: {} {}  : {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.layer)))
                
                # maybe not necessary
                self.apply_normalize_Wmats(ww_layer, params)
                self.apply_esd(ww_layer, params)
                
                if ww_layer.evals is not None:
                    self.apply_fit_powerlaw(ww_layer, params)
                    if params[MP_FIT]:
                        logger.debug("MP Fitting Layer: {} {} ".format(ww_layer.layer_id, ww_layer.name)) 
                        self.apply_mp_fit(ww_layer, random=False, params=params)

                    if params[DELTA_ES] and params[PLOT]:
                        logger.debug("Computing and Plotting Deltas: {} {} ".format(ww_layer.layer_id, ww_layer.name)) 
                        self.apply_plot_deltaEs(ww_layer, random=False, params=params)
                    
                    if params[VECTORS]:# and params[PLOT]:
                        logger.debug("Computing and Plotting Vector Localization Metrics: {} {} ".format(ww_layer.layer_id, ww_layer.name)) 
                        self.apply_analyze_eigenvectors(ww_layer, params=params)

                        
                    if params[RANDOMIZE]:# params['mp_fit']:
                        logger.debug("Randomizing Layer: {} {} ".format(ww_layer.layer_id, ww_layer.name))
                        self.apply_random_esd(ww_layer, params)
                        logger.debug("MP Fitting Random layer: {} {} ".format(ww_layer.layer_id, ww_layer.name)) 
                        self.apply_mp_fit(ww_layer, random=True, params=params)

                        if params[DELTA_ES] and params[PLOT]:
                            logger.debug("Cpmputing and Plotting Deltas: {} {} ".format(ww_layer.layer_id, ww_layer.name))
                            self.apply_plot_deltaEs(ww_layer, random=True, params=params)
                        
                    if params[DETX]:
                        logger.debug("Finding detX constaint: {} {} ".format(ww_layer.layer_id, ww_layer.name)) 
                        self.apply_detX(ww_layer, params=params)
                    
                    self.apply_norm_metrics(ww_layer, params)
                    #all_evals.extend(ww_layer.evals)
                    
                # TODO: add find correlation traps here

                # issue 137
                # details = details.append(ww_layer.get_row(), ignore_index=True)
                data = pd.DataFrame.from_records(ww_layer.get_row() , index=[0])
                details = pd.concat([details,data])

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
    @telly.count_decorator
    def describe(self, model=None, layers=[], min_evals=0, max_evals=None,
                min_size=None, max_size=None, 
                glorot_fix=False, 
                savefig=DEF_SAVE_DIR,
                conv2d_fft=False, conv2d_norm=True,  ww2x=False, 
                intra=False, channels=None, stacked=False, fix_fingers=False, start_ids=0):
        """
        Same as analyze() , but does not run the ESD or Power law fits
        
        """

        model = model or self.model    
        
        if min_size or max_size:
            logger.warning("min_size and max_size options changed to min_evals, max_evals, ignored for now")     

        params = DEFAULT_PARAMS
        params=DEFAULT_PARAMS
        params[MIN_EVALS] = min_evals 
        params[MAX_EVALS] = max_evals
      
        # params[NORMALIZE] = normalize  #removed 0.6.5 
        params[GLOROT_FIT] = glorot_fix
        params[CONV2D_NORM] = conv2d_norm
        params[CONV2D_FFT] = conv2d_fft
        params[WW2X] = ww2x   
        params[INTRA] = intra 
        params[CHANNELS_STR] = channels
        params[LAYERS] = layers
        params[STACKED] = stacked
        
        params[SAVEFIG] = savefig
        #params[SAVEDIR] = savedir
        params[START_IDS] = start_ids


        logger.info("params {}".format(params))
        if not self.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)

        layer_iterator = self.make_layer_iterator(model=model, layers=layers, params=params)            
        details = pd.DataFrame(columns=['layer_id', 'name'])
           
        num_all_evals = 0
        for ww_layer in layer_iterator:
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.debug("LAYER TYPE: {} {}  layer type {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.layer)))
                logger.debug("weights shape : {}  max size {}".format(ww_layer.weights.shape, params['max_evals']))
                if ww2x:
                    num_evals = ww_layer.M
                elif conv2d_fft:
                    num_evals = ww_layer.num_components
                else:
                    num_evals = ww_layer.M * ww_layer.rf
                    

                num_all_evals += num_evals    
                ww_layer.add_column('num_evals', num_evals)
                # issue 137
                #details = details.append(ww_layer.get_row(), ignore_index=True)
                data = pd.DataFrame.from_records(ww_layer.get_row() , index=[0])
                details = pd.concat([details,data])

        return details

    def valid_params(self, params):
        """Validate the input parametersm, return True if valid, False otherwise"""
        
        valid = True        
        xmin = params.get('xmin')

        if xmin and xmin not in [XMIN.UNKNOWN, XMIN.AUTO, XMIN.PEAK]:
            logger.warning("param xmin unknown, ignoring {}".format(xmin))
            valid = False
            
        xmax = params.get('xmax')
        if xmax and xmax not in [XMAX.UNKNOWN, XMIN.AUTO]:
            logger.warning("param xmax unknown, ignoring {}".format(xmax))
            valid = False
        
        min_evals = params.get('min_evals') 
        max_evals = params.get('max_evals')
        if min_evals and max_evals and min_evals >= max_evals:
            logger.warning("min_evals {} > max_evals {}".format(min_evals, max_evals))
            valid = False
        elif max_evals and max_evals < -1:
            logger.warning(" max_evals {} < -1 ".format(max_evals))
            valid = False
            
        # can not specify ww2x and conv2d_fft at same time
        if params.get(WW2X) and params.get('conv2d_fft'):
            logger.warning("can not specify ww2x and conv2d_fft")
            valid = False
            
            
        # can not specify intra and conv2d_fft at same time
        if params.get(INTRA) and params.get('conv2d_fft'):
            logger.warning("can not specify intra and conv2d_fft")
            valid = False
        
        # channels must be None, 'first', or 'last'
        channels = params.get(CHANNELS_STR) 
        if channels is not None and isinstance(channels,str):
            if channels.lower() != FIRST and channels.lower() != LAST:
                logger.warning("unknown channels {}".format(channels))
                valid = False

        # layer ids must be all positive or all negative
        filters = params.get(LAYERS) 
        if filters is not None:
            filter_ids = [int(f) for f in filters if type(f) is int]
          
            if len(filter_ids) > 0:
                if np.max(filter_ids) > 0 and np.min(filter_ids) < 0:
                    logger.warning("layer filter ids must be all > 0 or < 0: {}".format(filter_ids))
                    valid = False
         
        savefig = params.get(SAVEFIG)
        savedir = params.get(SAVEDIR)
        if savefig and isinstance(savefig,bool):
            logger.info("Saving all images to {}".format(savedir))
        elif savefig and isinstance(savefig,str):
            params[SAVEDIR] = savefig
            logger.info("Saving all images to {}".format(savedir))
        elif not isinstance(savefig,str) and not isinstance(savefig,bool):
            valid = False      
            
            
        fix_fingers =  params[FIX_FINGERS]
        if fix_fingers:
            if fix_fingers not in [XMIN_PEAK, CLIP_XMAX]:
                logger.warning("Unknown how to fix fingers {}, deactivating".format(fix_fingers))
            else:
                logger.info("Fixing fingers using  {}".format(fix_fingers))
                
            
        fit_type = params[FIT]
        if fit_type not in [PL, TPL, E_TPL, POWER_LAW, TRUNCATED_POWER_LAW]:
            logger.warning("Unknown fit type {}".format(fit_type))
            valid = False
            
        if fit_type==E_TPL and fix_fingers is not None:
            logger.warning("E-TPL set, fix_fingers being reset to XMIN_PEAK")


        intra = params[INTRA]
        if intra:
            if params[RANDOMIZE] or params[VECTORS]:
                logger.fatal("Can not set intra=True with randomize=True or vectors=True at this time")
                valid = False

        start_ids = params[START_IDS]
        if start_ids not in [0,1]:
            logger.fatal(f"Layer Ids must start at 0 or 1, start_ids={start_ids}")
            valid = False
                
        return valid
    
    
    def normalize_params(self, params):
        """Reset params to allow some syntatic sugar in the inputs"""
        
        fit_type = params[FIT]
        if fit_type==PL:
            params[FIT]=POWER_LAW
        elif fit_type==TPL:
            params[FIT]=TRUNCATED_POWER_LAW
        elif fit_type==E_TPL:
            params[FIT]=TRUNCATED_POWER_LAW
            params[FIX_FINGERS]=XMIN_PEAK
            
        # this may not work
        if params[CHANNELS_STR] and params[CHANNELS_STR] == FIRST:
             params[CHANNELS_STR]=CHANNELS.FIRST
        elif params[CHANNELS_STR] and params[CHANNELS_STR] == LAST:
             params[CHANNELS_STR]=CHANNELS.LAST
            
        return params
    
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

#     def pytorch_norm_fix(self, W, N, M, rf_size):
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
         
        normalize = params[NORMALIZE]
        glorot_fix = params[GLOROT_FIX]
        conv2d_norm = params[CONV2D_NORM]  # True
         
        all_evals = []

        logger.debug("generating {} replicas for each W of the random eigenvalues".format(num_replicas))
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
                sv = sp.linalg.svd(W, compute_uv=False)
                sv = sv.flatten()
                sv = np.sort(sv)[-n_comp:]    
                
                # sv = svd.singular_values_']
                
                if params[INTRA]:
                    evals = sv
                    sv = np.sqrt(sv)
                else:
                    evals = sv * sv 
                all_evals.extend(evals)
                                       
        return np.sort(np.array(all_evals))
   
    def plot_random_esd(self, ww_layer, params=DEFAULT_PARAMS):
        """Plot histogram and log histogram of ESD and randomized ESD"""
          
        savefig = params[SAVEFIG]
        savedir = params[SAVEDIR]

        layer_id = ww_layer.layer_id
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
        if savefig:
            #plt.savefig("ww.layer{}.esd.png".format(layer_id))
            save_fig(plt, "randesd1", layer_id, savedir)
        plt.show(); plt.clf()

        plt.hist(np.log10(nonzero_evals), bins=100, density=True, color='g', label='original')
        plt.hist(np.log10(nonzero_rand_evals), bins=100, density=True, color='r', label='random', alpha=0.5)
        plt.axvline(x=np.log10(max_rand_eval), color='orange', label='max rand')
        title = "Layer {} {}: Log10 ESD & Random ESD".format(ww_layer.layer_id,ww_layer.name)
        plt.title(title)   
        plt.xlabel(r"Log10 Eigenvalues $(log_{10}\lambda)$")               
        plt.legend()
        if savefig:
            #plt.savefig("ww.layer{}.randesd.2.png".format(layer_id))
            save_fig(plt, "randesd2", layer_id, savedir)
        plt.show(); plt.clf()
        

    def fit_powerlaw(self, evals, xmin=None, xmax=None, plot=True, layer_name="", layer_id=0, sample=False, sample_size=None, 
                     savedir=DEF_SAVE_DIR, savefig=True, svd_method=FULL_SVD, thresh=EVALS_THRESH, fix_fingers=False, fit_type=POWER_LAW):
        """Fit eigenvalues to powerlaw or truncated_power_lw
        
            if xmin is 
                'auto' or None, , automatically set this with powerlaw method
                'peak' , try to set by finding the peak of the ESD on a log scale
            
            if xmax is 'auto' or None, xmax = np.max(evals)
            
            svd_method = FULL_SVD (to add TRUNCATED_SVD with some cutoff)
            thresh is a threshhold on the evals, to be used for very large matrices with lots of zeros
            
                     
         """

        # when calling powerlaw methods, 
        # trap warnings, stdout and stderr 
        def pl_fit(data=None, xmin=None, xmax=None, verbose=False, distribution=POWER_LAW):
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f), warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=RuntimeWarning)
                return powerlaw.Fit(data, xmin=xmin, xmax=xmax, verbose=verbose, distribution=distribution, xmin_distribution=distribution)

        def pl_compare(fit, dist):
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f), warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=RuntimeWarning)
                return fit.distribution_compare(dist, TRUNCATED_POWER_LAW, normalized_ratio=True)
    
        status = None
        
        # defaults for failed status
        alpha = -1
        Lambda = -1
        D = -1
        sigma = -1
        xmin = -1  # not set / error
        xmax = None # or -1
        num_pl_spikes = -1
        best_fit = UNKNOWN
        fit = None
        
        # check    
        num_evals = len(evals)
        logger.debug("fitting {} on {} eigenvalues".format(fit_type, num_evals))

        if num_evals < MIN_NUM_EVALS:  # 10 , really 50
            logger.warning("not enough eigenvalues, stopping")
            status = FAILED
            return alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, best_fit, status
                          
        # if Power law, Lambda=-1 
        distribution = POWER_LAW
        if fit_type==TRUNCATED_POWER_LAW:
            distribution = 'truncated_power_law'
            
        
        # TODO: replace this with a robust sampler / estimator
        # requires a lot of refactoring below
        if sample and  sample_size is None:
            logger.info("setting sample size to default MAX_NUM_EVALS={}".format(MAX_NUM_EVALS))
            sample_size = MAX_NUM_EVALS
            
        if sample and num_evals > sample_size:
            logger.warning("samping not implemented in production yet")
            logger.info("chosing {} eigenvalues from {} ".format(sample_size, len(evals)))
            evals = np.random.choice(evals, size=sample_size)
                    
        if xmax == XMAX.AUTO or xmax is XMAX.UNKNOWN or xmax is None or xmax == -1:
            xmax = np.max(evals)
            
        if fix_fingers==XMIN_PEAK:
            logger.info("fix the fingers by setting xmin to the peak of the ESD")
            try:
                nz_evals = evals[evals > thresh]
                num_bins = 100  # np.min([100, len(nz_evals)])
                h = np.histogram(np.log10(nz_evals), bins=num_bins)
                ih = np.argmax(h[0])
                xmin2 = 10 ** h[1][ih]
                xmin_range = (0.95 * xmin2, 1.05 * xmin2)
                fit = pl_fit(data=nz_evals, xmin=xmin_range, xmax=xmax, verbose=False, distribution=distribution)  
                status = SUCCESS 
            except ValueError:
                status = FAILED
            except Exception:
                status = FAILED
                
        elif fix_fingers==CLIP_XMAX:
            logger.info("fix the fingers by fitting a clipped power law")
            try:
                fit = fit_clipped_powerlaw(evals, xmin=xmin, verbose=False)   
                status = SUCCESS 
            except ValueError:
                status = FAILED
            except Exception:
                status = FAILED
             
        elif xmin == XMAX.AUTO  or xmin is None or xmin == -1: 
            logger.debug("powerlaw.Fit no xmin , distribution={} ".format(distribution))
            try:
                nz_evals = evals[evals > thresh]
                fit = pl_fit(data=nz_evals, xmax=xmax, verbose=False, distribution=distribution)  
                status = SUCCESS 
            except ValueError:
                status = FAILED
            except Exception:
                status = FAILED

        else: 
            #logger.debug("POWERLAW DEFAULT XMIN SET ")
            try:
                fit = pl_fit(data=evals, xmin=xmin,  verbose=False, distribution=distribution)  
                status = SUCCESS 
            except ValueError:
                status = FAILED
            except Exception:
                status = FAILED
                    
        if fit is None or fit.alpha is None or np.isnan(fit.alpha):
            status = FAILED
            
        if status == FAILED:
            logger.warning("power law fit failed, will still attempt plots")
        else:
            alpha = fit.alpha 
            D = fit.D
            sigma = fit.sigma
            xmin = fit.xmin
            xmax = fit.xmax
            num_pl_spikes = len(evals[evals>=fit.xmin])
            if fit_type==TRUNCATED_POWER_LAW:
                alpha = fit.truncated_power_law.alpha
                Lambda = fit.truncated_power_law.Lambda
                
            logger.debug("finding best distribution for fit, TPL or other ?")
            # we stil check againsgt TPL, even if using PL fit
            all_dists = [TRUNCATED_POWER_LAW, POWER_LAW, LOG_NORMAL]#, EXPONENTIAL]
            Rs = [0.0]
            dists = [POWER_LAW]
            for dist in all_dists[1:]:
                R, p = pl_compare(fit, dist)
                if R > 0.1 and p > 0.05:
                    dists.append(dist)
                    Rs.append(R)
                    logger.debug("compare dist={} R={:0.3f} p={:0.3f}".format(dist, R, p))
            best_fit = dists[np.argmax(Rs)]
            
            # check status for over-trained, under-trained    
            # maybe should remove this
            if alpha < 2.0:
                status = OVER_TRAINED
            elif alpha > 6.0:
                status = UNDER_TRAINED
               

        if plot:
            
            if status==SUCCESS:
                fig2 = fit.plot_pdf(color='b', linewidth=0) # invisbile
                fig2 = fit.plot_pdf(color='r', linewidth=2)
                if fit_type==POWER_LAW:
                    fit.power_law.plot_pdf(color='r', linestyle='--', ax=fig2)
                else:
                    fit.truncated_power_law.plot_pdf(color='r', linestyle='--', ax=fig2)
            
            plot_loghist(evals[evals>(xmin/100)], bins=100, xmin=xmin)
            title = "Log-Log ESD for {}\n".format(layer_name) 
            title = title + r"$\alpha=${0:.3f}; ".format(alpha) + \
                r'$D_{KS}=$'+"{0:.3f}; ".format(D) + \
                r"$\lambda_{min}=$"+"{0:.3f}".format(xmin) + "\n"

            plt.title(title)
            plt.legend()
            if savefig:
                #plt.savefig("ww.layer{}.esd.png".format(layer_id))
                save_fig(plt, "esd", layer_id, savedir)
            plt.show(); plt.clf()
                
    
            # plot eigenvalue histogram
            num_bins = 100  # np.min([100,len(evals)])
            plt.hist(evals, bins=num_bins, density=True)
            title = "Lin-Lin ESD for {}".format(layer_name) 
            plt.title(title)
            plt.axvline(x=fit.xmin, color='red', label=r'$\lambda_{xmin}$')
            plt.legend()
            if savefig:
                #plt.savefig("ww.layer{}.esd2.png".format(layer_id))
                save_fig(plt, "esd2", layer_id, savedir)
            plt.show(); plt.clf()

            # plot log eigenvalue histogram
            nonzero_evals = evals[evals > 0.0]
            plt.hist(np.log10(nonzero_evals), bins=100, density=True)
            title = "Log-Lin ESD for {}".format(layer_name) 
            plt.title(title)
            plt.axvline(x=np.log10(fit.xmin), color='red', label=r'$\lambda_{xmin}$')
            plt.axvline(x=np.log10(fit.xmax), color='orange',  label=r'$\lambda_{xmax}$')
            plt.legend()
            if savefig:
                #plt.savefig("ww.layer{}.esd3.png".format(layer_id))
                save_fig(plt, "esd3", layer_id, savedir)
            plt.show(); plt.clf()
    
            # plot xmins vs D
            
            plt.plot(fit.xmins, fit.Ds, label=r'$D_{KS}$')
            plt.axvline(x=fit.xmin, color='red', label=r'$\lambda_{xmin}$')
            #plt.plot(fit.xmins, fit.sigmas / fit.alphas, label=r'$\sigma /\alpha$', linestyle='--')
            plt.xlabel(r'$x_{min}$')
            plt.ylabel(r'$D_{KS}$')
            title = r'$D_{KS}$' + ' vs.' + r'$x_{min},\;\lambda_{xmin}=$'
            plt.title(title+"{:0.3}".format(fit.xmin))
            plt.legend()
            if savefig:
                save_fig(plt, "esd4", layer_id, savedir)
                #plt.savefig("ww.layer{}.esd4.png".format(layer_id))
            plt.show(); plt.clf() 
            import sys, os

        return alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, best_fit, status
    
    @telly.count_decorator
    def get_ESD(self, model=None, layer=None, random=False, params=DEFAULT_PARAMS):
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
    

        layer_iter = WWLayerIterator(model=model, filters=[layer], params=params)     
        details = pd.DataFrame(columns=['layer_id', 'name'])
           
        ww_layer = next(layer_iter)
        assert(not ww_layer.skipped) 
        assert(ww_layer.has_weights)
        
        if not random:
            logger.info("Getting ESD for layer {} ".format(layer))
            self.apply_esd(ww_layer, params)
            esd = ww_layer.evals
        else:
            logger.info("Getting Randomized ESD for layer {} ".format(layer))
            self.apply_random_esd(ww_layer, params)
            esd = ww_layer.rand_evals

        if esd is None or len(esd)==0:
            logger.warning("No eigenvalues found for {} {}".format(ww_layer.layer_id, ww_layer.name))
                
        else:
            logger.debug("Found {} eiganvalues for {} {}".format(len(esd), ww_layer.layer_id, ww_layer.name))     
            
        return esd

    def get_Weights(self, model=None, layer=None, params=DEFAULT_PARAMS):
        """Get the Weights for the layer, specified by id or name)"""
        
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
    
        logger.info("Getting Weights for layer {} ".format(layer))

        layer_iter = WWLayerIterator(model=model, filters=[layer], params=params)     
        details = pd.DataFrame(columns=['layer_id', 'name'])
           
        ww_layer = next(layer_iter)
        assert(not ww_layer.skipped) 
        assert(ww_layer.has_weights)
        
        return ww_layer.Wmats
    
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

        N = ww_layer.N
        hard_rank = matrix_rank(np.sqrt(evals), N)
        entropy = matrix_entropy(np.sqrt(evals), N)
                    
        ww_layer.add_column(METRICS.NORM, norm)
        ww_layer.add_column(METRICS.LOG_NORM, log_norm)
        ww_layer.add_column(METRICS.SPECTRAL_NORM, spectral_norm)
        ww_layer.add_column(METRICS.LOG_SPECTRAL_NORM, log_spectral_norm)
        ww_layer.add_column(METRICS.ALPHA, alpha)
        ww_layer.add_column(METRICS.ALPHA_WEIGHTED, alpha_weighted)
        ww_layer.add_column(METRICS.LOG_ALPHA_NORM, log_alpha_norm)
        ww_layer.add_column(METRICS.STABLE_RANK, stable_rank)
        ww_layer.add_column(METRICS.MATRIX_RANK, hard_rank)
        ww_layer.add_column(METRICS.MATRIX_ENTROPY, entropy)

        return ww_layer
    
    
    # TODO: add x bulk max yellow line for bulk edge for random
    def apply_plot_deltaEs(self, ww_layer, random=False, params=DEFAULT_PARAMS):
        """Plot the deltas of the layer ESD, both in a sequence as a histogram (level statisitcs)"""
        layer_id = ww_layer.layer_id
        name = ww_layer.name or ""
        layer_name = "{} {}".format(layer_id, name)
        
        savefig = params[SAVEFIG]
        savedir = params[SAVEDIR]

        if random:
            layer_name = "{} Randomized".format(layer_name)
            title = "Layer {} W".format(layer_name)
            evals = ww_layer.rand_evals
            color='mediumorchid'
            bulk_max = ww_layer.rand_bulk_max
        else:
            title = "Layer {} W".format(layer_name)
            evals = ww_layer.evals
            color='blue'

        # sequence of deltas    
        deltaEs = np.diff(evals)
        logDeltaEs = np.log10(deltaEs)
        x = np.arange(len(deltaEs))
        eqn = r"$\log_{10}\Delta(\lambda)$"
        plt.scatter(x,logDeltaEs, color=color, marker='.')
        
        if not random:
            idx = np.searchsorted(evals, ww_layer.xmin, side="left")        
            plt.axvline(x=idx, color='red', label=r'$\lambda_{xmin}$')
        else:
            idx = np.searchsorted(evals, bulk_max, side="left")        
            plt.axvline(x=idx, color='red', label=r'$\lambda_{+}$')

        plt.title("Log Delta Es for Layer {}".format(layer_name))
        plt.ylabel("Log Delta Es: "+eqn)
        plt.legend()
        if savefig:  
            #plt.savefig("ww.layer{}.deltaEs.png".format(layer_id))         
            save_fig(plt, "deltaEs", layer_id, savedir)
        plt.show(); plt.clf()

        
        # level statistics (not mean adjusted because plotting log)
        plt.hist(logDeltaEs, bins=100, color=color, density=True)
        plt.title("Log Level Statisitcs for Layer {}".format(layer_name))
        plt.ylabel("density")
        plt.xlabel(eqn)
        plt.legend()
        if savefig:  
            #plt.savefig("ww.layer{}.level-stats.png".format(layer_id))         
            save_fig(plt, "level-stats", layer_id, savedir)
        plt.show(); plt.clf()

    def apply_mp_fit(self, ww_layer, random=True, params=DEFAULT_PARAMS):
        """Perform MP fit on random or actual random eigenvalues
        N/A yet"""

        layer_id = ww_layer.layer_id
        name = ww_layer.name or ""
        layer_name = "{} {}".format(layer_id, name)
        
        savefig = params[SAVEFIG]
        savedir = params[SAVEDIR]
        plot = params[PLOT]
        
        rescale = params[RESCALE] #should be True always
        
        if random:
            layer_name = "{} Randomized".format(layer_name)
            title = "Layer {} W".format(layer_name)
            evals = ww_layer.rand_evals
            color='mediumorchid'
        else:
            title = "Layer {} W".format(layer_name)
            evals = ww_layer.evals
            color='blue'

        N, M = ww_layer.N, ww_layer.M
        rf = ww_layer.rf

        num_spikes, sigma_mp, mp_softrank, bulk_min, bulk_max,  Wscale =  self.mp_fit(evals, N, M, rf, layer_name, layer_id, plot, savefig, savedir, color, rescale)
        
        if random:
            ww_layer.add_column('rand_num_spikes', num_spikes)
            ww_layer.add_column('rand_sigma_mp', sigma_mp)
            ww_layer.add_column('rand_mp_softrank', mp_softrank)
            ww_layer.add_column('rand_W_scale', Wscale)
            ww_layer.add_column('rand_bulk_max', bulk_max)
            ww_layer.add_column('rand_bulk_min', bulk_min)
        else:
            ww_layer.add_column('num_spikes', num_spikes)
            ww_layer.add_column('sigma_mp', sigma_mp)
            ww_layer.add_column(METRICS.MP_SOFTRANK, mp_softrank)
            ww_layer.add_column('W_scale', Wscale)
            ww_layer.add_column('bulk_max', bulk_max)
            ww_layer.add_column('bulk_min', bulk_min)
        return 

    def mp_fit(self, evals, N, M, rf, layer_name, layer_id, plot, savefig, savedir, color, rescale):
        """Automatic MP fit to evals, compute numner of spikes and mp_softrank"""
        
        Q = N/M        
        to_plot = evals.copy()

        Wscale=1.0
        if rescale:
            Wnorm = np.sqrt(np.sum(evals))
            ### issue #60 
            #Wscale = np.sqrt(N*rf)/Wnorm
            Wscale = np.sqrt(to_plot.shape[0])/Wnorm
            #logger.info("rescaling {} ESD of W by {:0.2f}".format(layer_id, Wscale))

        to_plot = (Wscale*Wscale)*to_plot
        lambda_max = np.max(to_plot)
        
        bw = 0.1 
        s1, f1 = fit_density_with_range(to_plot, Q, bw = bw)
        sigma_mp = s1
        
        bulk_max = (s1 * (1 + 1/np.sqrt(Q)))**2
        bulk_min = (s1 * (1 - 1/np.sqrt(Q)))**2
        
        #TODO: add Tracy Widom (TW) range
        #num_spikes = len(to_plot[to_plot > bulk_max])

        TW = 1/np.sqrt(Q)*np.power(bulk_max, 2/3)*np.power(M, -2/3)
        # Original "true" TW  should be divided by `np.power(Wscale, 2*2/3)`
        # Then, the new TW should be scaled by `(Wscale**2)**2 = np.power(Wscale, 4)`. This gives 8/3
        #
        # CHM  I dont think we need this
        TW_delta = TW#  U*np.power(Wscale, 8/3)
        bulk_max_TW = bulk_max + np.sqrt(TW_delta)
        
        logger.debug("bulk_max = {:0.3f}, bulk_max_TW = {:0.3f} ".format(bulk_max,bulk_max_TW))
        num_spikes = len(to_plot[to_plot > bulk_max_TW])
        
        #issue #69
        if M > 1:
            ratio_numofSpikes  = num_spikes / (M - 1)
        else:
            ratio_numofSpikes = 0

        mp_softrank = bulk_max / lambda_max

        if Q == 1.0:
            fit_law = 'QC SSD'
            
            #TODO: set cutoff 
            #Even if the quarter circle applies, still plot the MP_fit
            if plot:
                plot_density(to_plot, Q=Q, sigma=s1, method="MP", color=color, cutoff=bulk_max_TW)#, scale=Wscale)
                plt.legend([r'$\rho_{emp}(\lambda)$', 'MP fit'])
                plt.title("MP ESD, sigma auto-fit for {}".format(layer_name))
                if savefig:
                    #plt.savefig("ww.layer{}.mpfit1.png".format(layer_id))
                    save_fig(plt, "mpfit1", layer_id, savedir)
                plt.show(); plt.clf()
            
        else:
            fit_law = 'MP ESD'
#        
        #logger.info("MP fit min_esd={:0.2f}, max_esd={:0.2f}, Q={}, s1={:0.2f} Wsc ale={:0.2f}".format(np.min(to_plot), np.max(to_plot), Q, s1, Wscale))
        plot_density_and_fit(model=None, eigenvalues=to_plot, layer_name=layer_name, layer_id=0,
                              Q=Q, num_spikes=0, sigma=s1, verbose = False, plot=plot, color=color, cutoff=bulk_max_TW)#, scale=Wscale)
        
        if plot:
            title = fit_law+" for layer "+layer_name+"\n Q={:0.3} ".format(Q)
            title = title + r"$\sigma_{mp}=$"+"{:0.3} ".format(sigma_mp)
            title = title + r"$\mathcal{R}_{mp}=$"+"{:0.3} ".format(mp_softrank)
            title = title + r"$\#$ spikes={}".format(num_spikes)
            plt.title(title)
            if savefig:
                #plt.savefig("ww.layer{}.mpfit2.png".format(layer_id))
                save_fig(plt, "mpfit2", layer_id, savedir)
            plt.show(); plt.clf()
            
        bulk_max = bulk_max/(Wscale*Wscale)
        bulk_min = bulk_min/(Wscale*Wscale)
        return num_spikes, sigma_mp, mp_softrank, bulk_min, bulk_max, Wscale

        
    def smooth_W_alt(self, W, n_comp):
        """Apply the SVD Smoothing Transform to W"
        if n_comp < 0, then chomp off the top n_comp eiganvalues
        """       
        
        N, M = np.max(W.shape), np.min(W.shape)

        # TODO: replace this with truncated SVD
        # can't we just appky the svd transform...test
        # keep this old method for historical comparison
        u, s, vh = sp.linalg.svd(W, compute_uv=True)
                
        # s is ordered highest to lowest
        # i.e.  
        #    s = np.array([5,4,3,2,1])
        #
        # zero out all but the first n components
        # s[2:]   [3, 2, 1] 
        #
        # zero out the last n components
        # s[:2]   [5,4]
        #
        if n_comp > 0:
            s[n_comp:]=0.0  
        else:
            s[:-n_comp]=0.0
            
        s = list(s)
        s.extend([0]*(N-M))
        s = np.array(s)
        s = np.diag(s)
        if u.shape[0] > vh.shape[0]:
          smoothed_W = np.dot(np.dot(u,s)[:N,:M],vh)
        else:
          smoothed_W = np.dot(u, np.dot(s,vh)[:M,:N])
    
        return smoothed_W
    
    
 
    # these methods really belong in RMTUtil
    def smooth_W(self, W, n_comp):
        """Apply the sklearn TruncatedSVD method to each W, return smoothed W
        
        """
                
        svd = TruncatedSVD(n_components=n_comp, n_iter=7, random_state=42)
        if W.shape[0]<W.shape[1]:
            X = svd.fit_transform(W.T)
            VT = svd.components_
            smoothed_W = np.dot(X,VT).T     

        else:
            X = svd.fit_transform(W)
            VT = svd.components_
            smoothed_W = np.dot(X,VT)
        
        logger.debug("smoothed W {} -> {} n_comp={}".format(W.shape, smoothed_W.shape, n_comp))

        return smoothed_W
    
    
    #def clean_W(self, W):
    #    """Apply pyRMT RIE cleaning method"""
    #    
    #    return pyRMT.optimalShrinkage(W)

    @telly.count_decorator
    def SVDSmoothing(self, model=None, percent=0.2, ww2x=False, layers=[], method=SVD, fit=PL, plot=False, start_ids=0):
        """Apply the SVD Smoothing Transform to model, keeping (percent)% of the eigenvalues
        
        layers:
            List of layer ids. If empty, analyze all layers (default)
            If layer ids < 0, then skip the layers specified
            All layer ids must be > 0 or < 0
        
        ww2x:
            Use weightwatcher version 0.2x style iterator, which slices up Conv2D layers in N=rf matrices
            
        """
        
        model = model or self.model   
         
        params = DEFAULT_PARAMS
        
        params[WW2X] = ww2x
        params[LAYERS] = layers
        params[FIT] = fit # only useful for method=LAMBDA_MINa
        params[PLOT] = False
        params[START_IDS] = start_ids

        
        if ww2x:
            msg = "ww2x not supported yet for SVDSharpness, ending"
            logger.error(msg)
            raise Exception(msg)
        
        if method not in [SVD, RMT, DETX, LAMBDA_MIN]:
            logger.fatal("Unknown Smoothing method {}, stopping".format(method))
        else:
            logger.info(" Smoothing method {}".format(method))
            params[SMOOTH]=method
        
        # check framework, return error if framework not supported
        # need to access static method on  Model class

        logger.info("params {}".format(params))
        if not self.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)
     
        #TODO: restrict to ww2x or intra
        layer_iterator = self.make_layer_iterator(model=model, layers=layers, params=params)         
        
        # iterate over layers
        #   naive implementation uses just percent, not the actual tail
        #   we eventually want to compute the eigenvectors and analyze them
        #   here we do SVD
        
        for ww_layer in layer_iterator:
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.info("LAYER: {} {}  : {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.layer)))
                
                if method==LAMBDA_MIN:
                    self.apply_esd(ww_layer, params)
                    self.apply_fit_powerlaw(ww_layer, params)
                    params['num_smooth'] = ww_layer.num_pl_spikes
                elif method==DETX:
                    self.apply_esd(ww_layer, params)
                    self.apply_detX(ww_layer, params)
                    params['num_smooth'] = ww_layer.detX_num
                else:
                    params['num_smooth'] = int(percent*ww_layer.M*ww_layer.rf)
                    
                # TODO: do not recompute ESD if present ?
                self.apply_svd_smoothing(ww_layer, params)
        
        logger.info("Returning smoothed model")
        return model   

    
   
    
    # TODO: add methods that use ETPL-alpha and/or DETX
        
    def apply_svd_smoothing(self, ww_layer, params=DEFAULT_PARAMS):
        """run truncated SVD on layer weight matrices and reconstruct the weight matrices 
        keep all eigenvlues > percent*ncomp
        if percent < 0, then keep those < than percent*ncomp"""
        
        num_smooth = params['num_smooth']
      
        layer = ww_layer.layer
        layer_id = ww_layer.layer_id
        layer_name = ww_layer.name
        layer_type = ww_layer.the_type
        framework = ww_layer.framework
        channels= ww_layer.channels

        
        if framework not in [FRAMEWORK.KERAS, FRAMEWORK.PYTORCH, FRAMEWORK.ONNX]:
            logger.error("Sorry, SVDSmoothing does not support this model framework ")
            return 

        if channels == CHANNELS.UNKNOWN:
            log.error("Sorry, SVDSmoothing does not understand the channels for this layer, stopping ")
            return 
         
        M = ww_layer.M
        N = ww_layer.N
        rf = ww_layer.rf
        
        if params[SMOOTH]==RMT:
            logger.info("applying RMT method, ignoring num_smooth options")
        
        n_comp = num_smooth
        if num_smooth < 0:
            n_comp = M + num_smooth
            
        logger.info("apply truncated SVD on Layer {} {}, with nsmooth={},  keeping ncomp={} out of {}. of the singular vectors".format(layer_id, layer_name, num_smooth, n_comp, ww_layer.num_components))
                 
        # get the model weights and biases directly, converted to numpy arrays        
        has_W, old_W, has_B, old_B = ww_layer.get_weights_and_biases()
        
        logger.info("LAYER TYPE  {} out of {} {} {} ".format(layer_type,LAYER_TYPE.DENSE, LAYER_TYPE.CONV1D, LAYER_TYPE.EMBEDDING))          

        if layer_type in [LAYER_TYPE.DENSE, LAYER_TYPE.CONV1D, LAYER_TYPE.EMBEDDING]:
            if params[SMOOTH]==RMT:
                logger.fatal("RMT smoothing method removed")
                #new_W = self.clean_W(old_W) 
            elif num_smooth > 0:
                logger.debug("Keeping top {} singular values".format(num_smooth))
                new_W = self.smooth_W(old_W, num_smooth) 
            elif num_smooth < 0:
                logger.debug("Chomping off top {} singular values".format(-num_smooth))
                new_W = self.smooth_W_alt(old_W, num_smooth) 
            else:
                logger.warning("Not smoothing {} {}, ncomp=0".format(layer_id, layer_name))
                new_W  = old_W

            new_B = old_B
            # did we flip W when analyzing ?
            if new_W.shape != old_W.shape:
                new_W=new_W.T
                
            self.replace_layer_weights(framework, layer_id, layer, new_W, new_B)

                    
        # if not ww2x, then we need to divide num_smooth / rf   
        elif layer_type == LAYER_TYPE.CONV2D:                           
            new_W = np.zeros_like(old_W)
            new_B = old_B
            
            num_smooth = int(np.ceil(num_smooth/rf))
            
            if new_B is not None:
                logger.warning("Something went wrong, Biases found for Conv2D layer, layer {} {} ".format(layer_id, layer_name))
            
            #[k,k,M,N]
            if channels == CHANNELS.FIRST:
                i_max, j_max, _, _ = new_W.shape
                if rf != i_max*j_max:
                    logger.warning("channels FIRST not processed correctly W_slice.shape {}, rf={} ?".format(new_W.shape, rf))

                for i in range(i_max):
                    for j in range(j_max):                         
                        if num_smooth > 0:
                            logger.debug("Keeping top {} singular values".format(num_smooth))
                            new_W[i,j,:,:] = self.smooth_W(old_W[i,j,:,:], num_smooth)
                        elif num_smooth < 0:
                            logger.debug("Chomping off top {} singular values".format(-num_smooth))
                            new_W[i,j,:,:] = self.smooth_W_alt(old_W[i,j,:,:], num_smooth)
                        else:
                            new_W[i,j,:,:] = old_W[i,j,:,:]
                 
            #[N,M,k,k]
            elif channels == CHANNELS.LAST:
                _, _, i_max, j_max = new_W.shape
                if rf != i_max*j_max:
                    logger.warning("channels LAST not processed correctly W_slice.shape {}, rf={} ?".format(new_W.shape, rf))

                for i in range(i_max):
                    for j in range(j_max):   
                        if num_smooth > 0:
                            logger.debug("Keeping top {} singular values".format(num_smooth))
                            new_W[:,:,i,j] = self.smooth_W(old_W[:,:,i,j], num_smooth)
                        elif num_smooth < 0:
                            logger.debug("Chomping off top {} singular values".format(-num_smooth))
                            new_W[:,:,i,j] = self.smooth_W_alt(old_W[:,:,i,j], num_smooth)
                        else:
                            new_W[:,:,i,j] = old_W[:,:,i,j]
                        
            else:
                logger.warning("Something went wrong, channels not defined or detected for Conv2D layer, layer {} {} skipped ".format(layer_id, layer_name))
            
            self.replace_layer_weights(framework, layer_id, layer, new_W)
    

        else:
            logger.warning("Something went wrong,UNKNOWN layer {} {} skipped , type={}".format(layer_id, layer_name, layer_type))

        return ww_layer
        

    @telly.count_decorator
    def SVDSharpness(self, model=None,  ww2x=False, layers=[], plot=False, start_ids=0):
        """Apply the SVD Sharpness Transform to model
        
        layers:
            List of layer ids. If empty, analyze all layers (default)
            If layer ids < 0, then skip the layers specified
            All layer ids must be > 0 or < 0
        
        ww2x:
            Use weightwatcher version 0.2x style iterator, which slices up Conv2D layers in N=rf matrices
            
        """
        
        #TODO: check this
        model = model or self.model   
         
        params=DEFAULT_PARAMS
        params[WW2X] = ww2x
        params[LAYERS] = layers
        params[PLOT] = plot
        params[START_IDS] = start_ids

        if ww2x:
            msg = "ww2x not supported yet for SVDSharpness, ending"
            logger.error(msg)
            raise Exception(msg)
        
        # check framework, return error if framework not supported
        # need to access static method on  Model class

        logger.info("params {}".format(params))
        if not self.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)

     
        #TODO: restrict to ww2x or intra
        layer_iterator = self.make_layer_iterator(model=model, layers=layers, params=params)
            
        for ww_layer in layer_iterator:
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.info("LAYER: {} {}  : {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.layer)))
                self.apply_svd_sharpness(ww_layer, params)
        
        logger.info("Returning sharpened model")
        return model  
    
    
    
    def apply_svd_sharpness(self, ww_layer, params=DEFAULT_PARAMS):
        """run permute layer, run power law, identify and remove the spikes"""
        
        self.apply_permute_W(ww_layer, params)
        self.apply_esd(ww_layer, params)
        self.apply_mp_fit(ww_layer, random=False, params=params)

        params['num_smooth'] = - ww_layer.num_spikes
        logger.debug("Detected {} spikes".format(ww_layer.num_spikes))
        
        self.apply_svd_smoothing(ww_layer, params)
        self.apply_unpermute_W(ww_layer, params)

        return ww_layer


    # TODO: put this on the layer itself
    def replace_layer_weights(self, framework, idx, layer, W, B=None):
        """Replace the old layer weights with the new weights
        
        framework:  FRAMEWORK.KERAS | FRAMEWORK.PYTORCH
        
        layer: is the framework layerm, not an instance of WWLayer
        
        new_W:  numpy array 
        new_B:  numpy vector (array)
        
        
        """
        
        if framework==FRAMEWORK.KERAS:
            # (I think) this works for Dense and Conv2D, not sure about other layers
            if B is not None:
                W = [W, B]     
            layer.set_weights(W)
            
        elif framework==FRAMEWORK.PYTORCH:
            # see: https://discuss.pytorch.org/t/fix-bias-and-weights-of-a-layer/75120/4
            # this may be deprecated
            layer.weight.data = torch.from_numpy(W)
            if B is not None:
                layer.bias.data = torch.from_numpy(B)
                
        # See; https://github.com/onnx/onnx/issues/2978
        elif framework==FRAMEWORK.ONNX:
            #if B is not None:
            #    W = [W, B]   
            #else:
            #    W = [W]
            layer.set_weights(idx, W)
   
        else:
            logger.debug("Layer {} skipped, Layer Type {} not supported".format(layer_id, the_type))

        return
   

    def analyze_vectors(self, model=None, layers=[], min_evals=0, max_evals=None,
                plot=True,  savefig=DEF_SAVE_DIR, channels=None):
        """Seperate method to analyxe the eigenvectors of each layer
        
        ootions"""
        
        model = model or self.model   
        
        params=DEFAULT_PARAMS
        params[SAVEFIG] = savefig
        
        logger.debug("params {}".format(params))
        if not self.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        
        params = self.normalize_params(params)
        logger.info("params {}".format(params))

        layer_iterator = self.make_layer_iterator(model=model, layers=layers, params=params)
        
        for id, ww_layer in enumerate(layer_iterator):
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.info("LAYER: {} {}  : {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.layer)))
                self.apply_analyze_eigenvectors(ww_layer, params)
                
        return   


    def apply_analyze_eigenvectors(self, ww_layer, params=DEFAULT_PARAMS):
        """Compute and plot eigenvectors metrics: 

        - vector_entropies
        - localization_ratios
        - participation_ratios
        
        Note:  no  normalization is applied
        
        Does not modify the ww_layer (yet)
        
        """
        
        layer_id = ww_layer.layer_id
        name = ww_layer.name or ""
        layer_name = "{} {}".format(layer_id, name)

        print(f"SP {layer_name}")
        logger.warning(f"SP {layer_name}")

        
        M = ww_layer.M
        N = ww_layer.N    
        
        Wmats = ww_layer.Wmats
        if type(Wmats) is not list:
            Wmats = [Wmats]

        savedir = params.get(SAVEDIR)
        savefig = params.get(SAVEFIG)
        all_evals = []

        all_vec_entropies = []
        all_loc_ratios = []
        all_part_ratios = []
        
        for W in Wmats:
            W = W.astype(float)
            if W.shape[0]<=W.shape[1]:
                X = np.matmul(W, W.T)
            else:
                X = np.matmul(W.T, W)

            evals, V = sp.linalg.eig(X)
            all_evals.extend(evals)

            vec_entropies = []
            loc_ratios = []
            part_ratios = []
                
            for col in range(min(M,N)):
                vec_entropies.append(discrete_entropy(V[:,col]))
                loc_ratios.append(localization_ratio(V[:,col]))
                part_ratios.append(participation_ratio(V[:,col]))

            all_vec_entropies.extend(vec_entropies)
            all_loc_ratios.extend(loc_ratios)  
            all_part_ratios.extend(part_ratios)   
                
        
        sort_ids = np.argsort(all_evals)
                
        if params[PLOT]:
            fig, axs = plt.subplots(4)
            fig.suptitle("Vector Localization Metrics for {}".format(layer_name))   
            
            data = np.array(all_vec_entropies)[sort_ids]
            axs[0].scatter(np.arange(len(data)), data, marker=".", label='vec_entropy')
            axs[0].set_ylabel("Vector Entropies")        
            axs[0].label_outer()   
            
            data = np.array(all_loc_ratios)[sort_ids]        
            axs[1].scatter(np.arange(len(data)), data, marker=".", label='loc_ratio')
            axs[1].set_ylabel("Localization Ratios")            
            axs[1].label_outer()   
            
            data = np.array(all_part_ratios)[sort_ids]        
            axs[2].scatter(np.arange(len(data)), data, marker=".", label='part_ratio')
            axs[2].set_ylabel("Participation Ratios")  
            axs[2].label_outer()     
            
            data = np.array(all_evals)[sort_ids]        
            axs[3].scatter(np.arange(len(data)), data, marker=".", label='eigenvalue')
            axs[3].set_ylabel("Eigenvalues")  
            axs[3].label_outer() 
        
            sorted_evals = np.array(all_evals)[sort_ids]        
            if ww_layer.has_column('xmin'):
                xmin = ww_layer.xmin
                #find index of eigenvalue closest to xmin
                
                xvals = np.where(sorted_evals < xmin)[0]
                if len(xvals)>0:
                    xval = xvals[-1]
                    for ax in axs:
                        ax.axvline(x=xval, color='r', label='xmin')
                        ax.legend()
                else:
                    logger.warning("xmin can not be displayed")
        
            if savefig:
                save_fig(plt, "vector_metrics", ww_layer.layer_id, savedir)
            plt.show(); plt.clf()

        # Histogram plots, similar to the JMLR paper
        if ww_layer.has_column('xmin'):
            xmin = ww_layer.xmin
            #find index of eigenvalue closest to xmin
            sorted_evals = np.array(all_evals)[sort_ids]        
            bulk_ids = np.where(sorted_evals < xmin)[0]
            tail_ids = np.where(sorted_evals >= xmin)[0]
            
            
            if len(bulk_ids)==0:
                logger.warning("no bulk data to plot, xmin={:0.2f}".format(xmin))
            if len(tail_ids)==0:
                logger.warning("no tail data to plot, xmin={:0.2f}".format(xmin))

            arrays = [all_vec_entropies, all_loc_ratios, all_part_ratios]
            titles = ["Vector Entropies", "Localization Ratios", "Participation Ratios"]
            names  = ['vec_entropy', 'loc_ratio', 'part_ratio']
            
            #layer stats
            for name, arr in zip(names,arrays):     
                data = np.array(arr)[sort_ids]        
                bulk_data = data[bulk_ids]
                tail_data = data[tail_ids]
                
                bulk_mean, bulk_var = np.mean(bulk_data), np.var(bulk_data)
                tail_mean, tail_var = np.mean(tail_data), np.var(tail_data)

                ww_layer.add_column("bulk_mean_{}".format(name), bulk_mean)
                ww_layer.add_column("bulk_var_{}".format(name), bulk_var)
 
                ww_layer.add_column("tail_mean_{}".format(name), tail_mean)
                ww_layer.add_column("tail_var_{}".format(name), tail_var)
                    

            if params[PLOT]:
                fig, axs = plt.subplots(3)
                fig.suptitle("Vector Bulk/Tail Metrics for {}".format(layer_name))   
                
                arrays = [all_vec_entropies, all_loc_ratios, all_part_ratios]
                titles = ["Vector Entropies", "Localization Ratios", "Participation Ratios"]
                
                for ix, ax in enumerate(axs):
                    arr = arrays[ix]
                    title = titles[ix]
                    
                    data = np.array(arr)[sort_ids]        
                    bulk_data = data[bulk_ids]
                    tail_data = data[tail_ids]
                    
                    
                    # should never happen
                    if len(bulk_data)>0:
                        ax.hist(bulk_data, bins=100, color='blue', alpha=0.5, label='bulk', density=True)
    
                    # might happen
                    if len(tail_data) > 0:
                        ax.hist(tail_data, bins=100, color='purple', alpha=0.5, label='tail', density=True)
    
                    ax.set_ylabel(title) 
                    ax.label_outer() 
                    ax.legend()
                    
                    
                
                if savefig:
                    save_fig(plt, "vector_histograms", ww_layer.layer_id, savedir)
                plt.show(); plt.clf()
       
       
        return

        
