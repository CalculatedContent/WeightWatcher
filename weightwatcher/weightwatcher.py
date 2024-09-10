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
import glob, json
import traceback
import tempfile

#from deprecated import deprecated
import inspect

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
#from sklearn.decomposition import TruncatedSVD

from copy import deepcopy
import importlib
import numbers

import safetensors
from safetensors import safe_open


#
# this is use to allow editing in Eclipse but also
# building on the commend line
# see: https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
#

from .RMT_Util import *
from .constants import *
from .WW_powerlaw import *


# WW_NAME moved to constants.py

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(WW_NAME) 
logger.setLevel(logging.WARNING)


mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)



def main():
    """
    Weight Watcher
    """
    print("WeightWatcher command line support coming later. https://calculationconsulting.com")


# TODO: make all these methods abstract
# can't do this until all the class methods are implemented and tested

class FrameworkLayer:
    """Base class for all classes that wrap the layer from each Framework and Format
    Each FrameworkLayer is specifically typed to make it easier to manage the different and growing
    """
    
    def __init__(self, layer, layer_id, name, longname="", weights=None, bias=None, 
                 the_type = LAYER_TYPE.UNKNOWN, skipped=False, framework=FRAMEWORK.UNKNOWN, 
                 channels=CHANNELS.UNKNOWN, plot_id=None, has_bias=False, lazy=False):
        
        self.lazy = lazy
        self.layer = layer
        self.layer_id = layer_id
        
        # read weights and biases
        self.name = name
        self.longname =  longname
        self.the_type = the_type
        self.skipped = skipped
        self.framework = framework
        self.channels = channels
        self.has_bias = has_bias
        
        
        if plot_id is None:
            self.plot_id = f"{layer_id}"
        else:
            self.plot_id = plot_id
            
            
        if self.name is None and hasattr(self.layer, 'name'):
            self.name = self.layer.name
        elif self.name is None:
            self.name = str(self.layer)
            self.name = re.sub(r'\(.*', '', self.name)
            
        if self.longname is None and hasattr(self.layer, 'longname'):
            self.longname = self.layer.longname
        elif self.longname is None:
            self.longname = name
            
            
    def layer_type(self, layer):
        """Given a framework layer, determine the weightwatcher LAYER_TYPE"""
        
        the_type = LAYER_TYPE.UNKNOWN
        typestr = (str(type(layer))).lower()     
          
            
        return the_type
    
    
    @staticmethod
    def get_layer_iterator(model):
        """should return an interator over the layer, that builds the subclass object"""
        pass

    #@abc.abstractmethod:=
    def has_biases(self):
        return self.has_bias
    
    #@abc.abstractmethod:
    def get_weights_and_biases(self):
        """   return has_weights, weights, has_biases, biases  """
        pass

    #@abc.abstractmethod:
    def replace_layer_weights(self, W, B=None):
        pass

    
    

class KerasLayer(FrameworkLayer):

    
    def __init__(self, layer, layer_id, name=None, longname = None):

        the_type = self.layer_type(layer)
        channels = CHANNELS.FIRST
        FrameworkLayer.__init__(self, layer, layer_id, name, longname=longname, the_type=the_type, 
                                framework=FRAMEWORK.KERAS, channels=channels)

    
    def layer_type(self, layer):
        """Given a framework layer, determine the weightwatcher LAYER_TYPE
        This can detect basic Keras  classes by type, and will try to infer the type otherwise. """
    

        
        the_type = LAYER_TYPE.UNKNOWN
        typestr = (str(type(layer))).lower()     
        
        # Keras TF 2.x types
        if isinstance(layer, keras.layers.Dense) or 'dense' in typestr:
           the_type = LAYER_TYPE.DENSE
           
        elif isinstance(layer, keras.layers.Conv1D)  or  'conv1d' in typestr:               
           the_type = LAYER_TYPE.CONV1D
        
        elif isinstance(layer, keras.layers.Conv2D) or 'conv2d' in typestr:             
           the_type = LAYER_TYPE.CONV2D
        #
        # elif isinstance(layer, keras.layers.Bidirectional) or 'bidirectional' in typestr:             
        #    the_type = LAYER_TYPE.BIDIRECTIONAL       
        #

                           
        elif isinstance(layer, keras.layers.Flatten) or 'flatten' in typestr:
           the_type = LAYER_TYPE.FLATTENED
           
        elif isinstance(layer, keras.layers.Embedding) or 'embedding' in typestr:
           the_type = LAYER_TYPE.EMBEDDING
           
        elif isinstance(layer, tf.keras.layers.LayerNormalization) or 'layernorn' in typestr:
           the_type = LAYER_TYPE.NORM
           
        return the_type
        

 
    def has_biases(self):
        return hasattr(self.layer, 'use_bias') and self.layer.use_bias is True

        #return self.layer.use_bias is True
    
         
    def get_weights_and_biases(self):
        """extract the original weights (as a tensor) for the layer, and biases for the layer, if present
        
        these wil be set in the enclosing WWLayer
        """
         
        has_weights, has_biases = False, False
        weights, biases = None, None
                   
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
            has_weights = True
            if self.has_biases():
                biases = w[1]
                has_biases = True
                
                
       ### BIDIRECTIONAL added as a hack not fuly tested
       # A bidirectional model has 4 distinct matrices,
       # each one would need to be treated as a different layer in keras
       # 
#        elif self.the_type==LAYER_TYPE.BIDIRECTIONAL:
#             weights = [w[0], w[1], w[3], w[4]]
#             has_weights = True
#             has_biases = False
##             if self.has_biases():
##                biases = [w[2], w[6]]
##                has_biases = True
#             
        else: 
            logger.info("keras layer: {} {}  type {} not found ".format(self.layer.name,str(self.layer),str(self.the_type)))
    
        return has_weights, weights, has_biases, biases  


    # warningL may not work for BIDIRECTIONAL
    def replace_layer_weights(self, W, B=None):
        """My not work,, see https://stackoverflow.com/questions/51354186/how-to-update-weights-manually-with-keras"""
        if self.has_biases() and B is not None:
            W = [W, B]
        self.layer.set_weights(W)
            
            
    @staticmethod
    def get_layer_iterator(model, start_id=0):
        """ start_id is 0 for back compatbility"""
        layer_id = start_id
        def layer_iter_():
            def traverse_(layer):
                "not recursive, just iterate over all submodules if present"
                nonlocal layer_id 
                if not hasattr(layer, 'submodules') or len(layer.submodules)==0:
                    keras_layer = KerasLayer(layer, layer_id)
                    layer_id += 1
                    yield keras_layer
                else:                        
                    for sublayer in layer.submodules:
                        keras_layer = KerasLayer(sublayer, layer_id)
                        layer_id += 1
                        yield keras_layer
            for layer in model.layers:
                yield from traverse_(layer)

        return layer_iter_() 
    

      
class PyTorchLayer(FrameworkLayer):
    
    def __init__(self, layer, layer_id, name=None, longname = None):
        
        the_type = self.layer_type(layer)
        channels = CHANNELS.LAST
        FrameworkLayer.__init__(self, layer, layer_id, name, longname=longname, the_type=the_type, 
                                framework=FRAMEWORK.PYTORCH, channels=channels)        
    
    def layer_type(self, layer):
        """Given a framework layer, determine the weightwatcher LAYER_TYPE
        This can detect basic  PyTorch classes by type, and will try to infer the type otherwise. """
        
        the_type = LAYER_TYPE.UNKNOWN
        typestr = (str(type(layer))).lower()     
         
        if isinstance(layer, torch.nn.Linear) or 'linear' in typestr:
             the_type = LAYER_TYPE.DENSE
             
        elif isinstance(layer, torch.nn.Conv1d) or  'conv1d' in typestr:
             the_type = LAYER_TYPE.CONV1D
         
        elif isinstance(layer, torch.nn.Conv2d) or 'conv2d' in typestr:
             the_type = LAYER_TYPE.CONV2D
             
        elif isinstance(layer, torch.nn.Embedding) or 'embedding' in typestr:
             the_type = LAYER_TYPE.EMBEDDING
         
        elif  'norm' in str(type(layer)).lower() :
             the_type = LAYER_TYPE.NORM

           
        return the_type
        
        
    def has_biases(self):
        return hasattr(self.layer, 'bias') and self.layer.bias is not None and self.layer.bias.data is not None
    
    def get_weights_and_biases(self):
        """extract the original weights (as a tensor) for the layer, and biases for the layer, if present
        
        expects self.layer to be set   
        """
         
        has_weights, has_biases = False, False
        weights, biases = None, None
        
        if hasattr(self.layer, 'weight'): 
            #w = [np.array(self.layer.weight.data.clone().cpu())]
            w = [torch_T_to_np(self.layer.weight.data)]
            
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
                has_weights = True
                
                biases = None
                has_biases = False
                if self.has_biases():
                    #biases = self.layer.bias.data.clone().cpu()
                    #biases = biases.detach().numpy()
                    biases = torch_T_to_np(self.layer.bias.data)            
                    has_biases = True

                
            elif self.the_type not in [LAYER_TYPE.NORM]: 
                logger.info("pytorch layer: {}  type {} not found ".format(str(self.layer),str(self.the_type)))
            else:
                pass           
        
        return has_weights, weights, has_biases, biases  
    
    
    def replace_layer_weights(self, W, B=None):
            
        self.layer.weight.data = torch.from_numpy(W)
        if self.has_biases() and B is not None:
            self.layer.bias.data = torch.from_numpy(B)
        
        
    
    @staticmethod
    def get_layer_iterator(model,start_id=0):
        """ start_id is 0 for back compatbility"""
        def layer_iter_():
            #for layer in model.modules():
            layer_id = start_id
            for longname, layer in model.named_modules():
                setattr(layer, 'longname', longname)
                pytorch_layer = PyTorchLayer(layer, layer_id, longname=longname)  
                layer_id += 1 
                yield pytorch_layer                 
        return layer_iter_()     
    
    
      
        


class PyStateDictLayer(FrameworkLayer):  
    """Similar to the PyTorch iterator, but the layer ids may be different"""
    
    def __init__(self, model, layer_id, name):
            
        self.model = model  # model_state_dict
        self.layer = name

        the_type = self.layer_type(self.layer)
        FrameworkLayer.__init__(self, name, layer_id, name, longname=name, the_type=the_type, 
                                framework=FRAMEWORK.PYSTATEDICT, channels=CHANNELS.LAST) 
        
        
    def has_biases(self):
        bias_key = self.layer + '.bias'
        if bias_key in self.model:
            return True
        return False
    
  
    def layer_type(self, layer):
        """Given a framework layer, determine the weightwatcher LAYER_TYPE"""
        
        the_type = LAYER_TYPE.UNKNOWN
          
        has_weights, weights, has_biases, biases = self.get_weights_and_biases()
        
        if len(weights)<2:
            pass
        elif len(weights.shape)==2:
            the_type = LAYER_TYPE.DENSE
        elif len(weights.shape)==4:
            the_type = LAYER_TYPE.CONV2D
        
        return the_type
    

    
    @staticmethod                  
    def get_layer_iterator(model_state_dict, start_id=1, layer_map=None):
        
        from copy import deepcopy
        """model is just a dict, but we need the name of the dict
        
        start_id = 0 is NOT ok since all counting starts at 1 for this layer
        
        layer_map = ordered list of keys (layer names in the file)"""
                        
        def layer_iter_(model_state_dict, layer_map):
            layer_id = start_id 
            
            if layer_map is None or len(layer_map)==0:
               layer_map = model_state_dict.keys()
               
            has_weight = any(key.endswith(".weight") for key in layer_map)
            if not has_weight:
                layer_names = [f"{x}.weight" for x in layer_map]
            else:
                layer_names = layer_map
                 
            for key in layer_names:
                
                # Check if the key corresponds to a weight matrix
                if key.endswith('.weight'):
                    # Extract the weight matrix and layer name
                    weights = model_state_dict[key]
                    layer_name = key[:-len('.weight')]
                    # Check if the layer has a bias vector
                    bias_key = layer_name + '.bias'
                    if bias_key in model_state_dict:
                        biases = model_state_dict[bias_key]
                    else:
                        biases = None
            
                    if type(weights)==torch.Tensor:
                        """We want to store data in float16, not 32"""
                        weights = torch_T_to_np(weights.data)
                        if biases is not None:
                            biases = torch_T_to_np(biases.data)
            
                        
                    # we may need to change this, set valid later
                    # because we want al the layers for describe
                    if weights is not None:
                        the_layer = PyStateDictLayer(model_state_dict, layer_id, layer_name)

                        layer_id += 1 # because we always start at 1 , we increment this after, not before
                        yield the_layer

        return layer_iter_(model_state_dict, layer_map)
    
    
    def get_weights_and_biases(self):
        """   return has_weights, weights, has_biases, biases  """
        
        model_state_dict = self.model
        weight_key = self.layer+'.weight'
        bias_key = self.layer+'.bias'

        weights = model_state_dict[weight_key]
        biases = None
        if self.has_biases():
            biases = model_state_dict[bias_key]

        if type(weights)==torch.Tensor:
            weights = torch_T_to_np(weights.data)
            if self.has_biases():
                biases = torch_T_to_np(biases.data)
                
                    
        return True, weights, self.has_biases(), biases
    
    
    def replace_layer_weights(self, W, B=None):
        """ replace weights and biases in the underlying layer
        
        expects to replace with torch arrays
        """
        
        model_state_dict = self.model
        weight_key = self.layer+'.weight'
        bias_key = self.layer + '.bias'
        
        model_state_dict[weight_key] = torch.from_numpy(W)
        if self.has_biases() and B is not None:
            model_state_dict[bias_key] = torch.from_numpy(B)
        
        return 

        
      
class PyStateDictDir(PyStateDictLayer):
    """Class that reads a list of pyStateDict .bin files (or .safetensors) 
    specified by the directory
    
       Expect formats:  pytorch_model.#.bin | model.#.safetensors
       
       i.e:
           pytorch_model.00001-of-00014.bin
           pytorch_model.00002-of-00014.bin
            ...
       
        and/or
       
           model.00001-of-00014.safetensors
           model.00002-of-00014.safetensors
            ...
            
        If the directory contains both, will selectively pick the safetensors first
        
        This will replace apply_watcher_to_pytorch_bins...
        
        Note: this is only used as a static class
        
    """

    
    @staticmethod
    def get_layer_map(fileglob): 
                
        # safetensors keys are stored in sort order, not layer order, so we need the ordering
        # so we either need to 
        # - read a custom layer_map file for each safetensors file, or
        # - embed the mapping in the safetensors metadata 
        # - read the huggingface model.safetensors.index.json file


        # Assumes there is 1 filename
        layer_map = []

        layer_map_filename = None
        filenames = glob.glob(fileglob.replace("safetensors", "layer_map"))
        if len(filenames)==1:
            layer_map_filename = filenames[0]
        elif len(filenames)>1:
            logger.warning(f"More than 1  layer_map_filename found! {filenames}")
            
        weight_map = []
        weight_map_filename = None
        filenames = glob.glob(fileglob.replace("safetensors", ".safetensors.index.json"))
        if len(filenames)==1:
            weight_map_filename = filenames[0]
        elif len(filenames)>1:
            logger.warning(f"More than 1  weight_map_filename found! {filenames}")
            

        # read layer map if found, otherwise ignore
        if layer_map_filename is not None and os.path.exists(layer_map_filename):
            logger.info(f"loading layer_map {layer_map_filename}")
            
            with open(layer_map_filename, 'r') as f:
                layer_map = [line.strip() for line in f]
                
            if len(layer_map) == 0:
                logger.critical(f"no layers found in {layer_map_filename}")
                
        elif weight_map_filename is not None and  os.path.exists(weight_map_filename):
            logger.info(f"loading weight_map {weight_map_filename}")
            
            with open(weight_map_filename, 'r') as f:
                data = json.load(f)
                weight_map = data['weight_map']
                files = np.unique([x for x in weight_map.values()])
                for file in files:
                    for k, v in weight_map.items():
                        if v == file:
                            layer_map.append(k)
                
        # read layer map if found, otherwise ignore
        elif layer_map_filename is not None and os.path.exists(layer_map_filename):
            logger.info(f"loading layer_map {layer_map_filename}")
            
            with open(layer_map_filename, 'r') as f:
                layer_map = [line.strip() for line in f]
                
            if len(layer_map) == 0:
                logger.critical(f"no layers found in {layer_map_filename}")
        else:
            # TODO just ignore state, buld as it goes
            logger.info(f"loading {layer_map_filename} not found, ignoring")
            
        return layer_map

    
    @staticmethod
    def read_safetensor_state_dict(state_dict_filename):
        """Reads the entire state dict into memory
        
        NOT USED YET """
        state_dict = {}
        with safe_open(state_dict_filename, framework="pt", device='cpu') as f:
            for layer_name in f.keys():
                state_dict[layer_name] = f.get_tensor(layer_name)
        return state_dict
        

    
    
    @staticmethod
    def get_layer_iterator(model_dir, start_id=1, layer_map=None):
        # TODO:  if layer_map is set, use this ordered
        # TODO:  find layer in the list of layers provided, skip if layer not found
        #        notice: we could use layer_map from the other model
        #        need to skip layers if found

        logger.debug(f"Buidling Layer for {model_dir}")    
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
            logger.fatal(f"Directory {model_dir} not found")    
            
        format, fileglob  = WeightWatcher.infer_model_file_format(model_dir)  

        if format not in [MODEL_FILE_FORMATS.SAFETENSORS, MODEL_FILE_FORMATS.PYTORCH]:
             logger.fatal(f"Unknown or unsupporteed model format {format}" )    
    
        
        def layer_iter_(fileglob, layer_map=None):
            # Track the number of layers already yielded
            num_layers_yielded = 0
            current_start_id = start_id  # Create a copy of start_id
                        
            if layer_map is None and format==MODEL_FILE_FORMATS.SAFETENSORS:    
                logger.info(f"Reading layer map from {fileglob}") 
                layer_map = PyStateDictDir.get_layer_map(fileglob)
            else:
                logger.info(f"Using specified layer map")
                            
            # loop over stat dict files in sort order
            
            # TODO: open all safetensors files at once
            # access layers as requested from map 
            
            if (layer_map is not None and len(layer_map)>0) and format==MODEL_FILE_FORMATS.SAFETENSORS:
                fileglob = f"{model_dir}/*model*safetensors"
                safetensors_dict = SafeTensorDict(fileglob)
                            
                current_start_id += num_layers_yielded 
                sub_layer_iter = PyStateDictLayer.get_layer_iterator(safetensors_dict, start_id = current_start_id, layer_map=layer_map)
                for layer in sub_layer_iter:
                    num_layers_yielded += 1  # Increment the counter
                    yield layer
                        
            
            elif (layer_map is None or len(layer_map)==0) and format==MODEL_FILE_FORMATS.SAFETENSORS:
                for state_dict_filename in sorted(glob.glob(fileglob)):
                    logger.info(f"loading {state_dict_filename}")
                    state_dict = SafeTensorDict(state_dict_filename)
                    #with safe_open(state_dict_filename, framework="pt", device='cpu') as f:
                    #    for layer_name in f.keys():
                    #        state_dict[layer_name] = f.get_tensor(layer_name)
                    #logger.info(f"Read safetensors file: {state_dict_filename}, len={len(state_dict)}")

                    # Yield individual layers
                    current_start_id += num_layers_yielded 
                    sub_layer_iter = PyStateDictLayer.get_layer_iterator(state_dict, start_id = current_start_id)
                    for layer in sub_layer_iter:
                        num_layers_yielded += 1  # Increment the counter
                        yield layer
                        
            else:  # format is PYTORCH BIN 
                fileglob = f"{model_dir}/*model*bin"
                for state_dict_filename in sorted(glob.glob(fileglob)):
                    logger.info(f"loading {state_dict_filename}")
                    
                    # is this correct ?
                    state_dict = torch.load(state_dict_filename, map_location=torch.device('cpu'))
                    logger.info(f"Read pytorch model bin file: {state_dict_filename}, len={len(state_dict)}")
                    
                     # Yield individual layers
                    current_start_id += num_layers_yielded
                    sub_layer_iter = PyStateDictLayer.get_layer_iterator(state_dict, start_id = current_start_id)
                    for layer in sub_layer_iter:
                        num_layers_yielded += 1  # Increment the counter
                        yield layer
                                                            
            
            # for state_dict_filename in sorted(glob.glob(fileglob)):
            #     logger.info(f"loading {state_dict_filename}")
            #
            #     if format==MODEL_FILE_FORMATS.SAFETENSORS:
            #         from safetensors import safe_open
            #
            #         #state_dict = {k: f.get_tensor(k) for k in safe_open(state_dict_filename, framework="pt", device='cpu').keys()}
            #         state_dict = {}
            #         with safe_open(state_dict_filename, framework="pt", device='cpu') as f:
            #             if layer_map is not None and len(layer_map)>0:
            #                 if not set(f.keys()).issubset(not compatble set(layer_map)):
            #                     logger.critical(f"safetensors file has keys not found in the layer_map!")
            #                 else:
            #                     for layer_name in f.keys():
            #                         state_dict[layer_name] = f.get_tensor(layer_name)
            #             #else:
            #                # 
            #
            #         logger.debug(f"Read safetensors: {state_dict_filename}, len={len(state_dict)}")
            #
            #     else:
            #         state_dict = torch.load(state_dict_filename, map_location=torch.device('cpu'))
            #         logger.debug(f"Read pytorch model bin file: {state_dict_filename}, len={len(state_dict)}")
            #
            #     # Update the start_id based on the number of layers already yielded
            #     current_start_id += num_layers_yielded
            #
            #     # Yield individual layers
            #     sub_layer_iter = PyStateDictLayer.get_layer_iterator(state_dict, current_start_id)
            #     for layer in sub_layer_iter:
            #         num_layers_yielded += 1  # Increment the counter
            #         yield layer
                                                             
        return layer_iter_(fileglob, layer_map=layer_map)
    
         
    
      
class WWFlatFile(FrameworkLayer):
    """Helper class to support layers directly from pyTorch StateDict
        
      Currently only supports DENSE layers: need to update
      
      initializer reads we logger.info(f"CReating CONFIG {config['layers'][3}")ights and bias file directly from disk
      
      would like to adapt to pystatedict to read off file
      we should let the user specify, and then it will create the temp files automatically ?
                
    """
    
    def __init__(self, layer_id, config, layer_config):
        self.config = config
        weights_dir =  config['weights_dir']
        
        self.layer_config = layer_config
        self.layer_id = int(layer_id)
        
        # read weights and biases
        name = layer_config['name']
        longname = layer_config['longname']
        the_type = LAYER_TYPE.UNKNOWN
                
        self.weights = None
        self.weightfile = layer_config['weightfile']
        self.weightfile = os.path.join(weights_dir, self.weightfile)
        self.has_weights = True

        self.bias = None
        self.biasfile = None
        self.has_bias = False
        if layer_config['biasfile']:
            self.biasfile = layer_config['biasfile']
            self.biasfile = os.path.join(weights_dir, self.biasfile)
            self.has_bias = True
        
        str_type = layer_config['type']
        the_type = WeightWatcher.layer_type_from_str(str_type)
          
        dims = layer_config['dims']
        dims = json.loads(dims)
        if the_type in [LAYER_TYPE.DENSE, LAYER_TYPE.NORM, LAYER_TYPE.CONV1D]:
            self.N = np.max(dims)
            self.M = np.min(dims)
            self.rf = 1
            self.dims = dims
            self.weight_dims = dims
            self.skipped = False
        elif the_type in [LAYER_TYPE.CONV2D]:
            # assume channel is not switched
            self.N = np.max(dims[0:2])
            self.M = np.min(dims[0:2])
            self.rf = dims[2]*dims[3]
            self.dims = dims
            self.weight_dims = dims
            self.skipped = False
        else:
            self.skipped = True
            logger.fatal
            (f"Sorry, WWFlatFile only supports DENSE and NORM layers currently, not {str_type}")
        
        
        FrameworkLayer.__init__(self, layer_config, int(layer_id), name, longname=longname, weights=None, bias=None, the_type=the_type, 
                                framework=FRAMEWORK.WW_FLATFILES, channels=CHANNELS.LAST, has_bias=self.has_bias, lazy=True)
        
        return 
    
    
    def lazy_load_weights_and_biases(self):
        """load the weights and biases from the flat files"""

        if self.weights is None:
            logger.info(f"Loading weights from {self.weightfile}")
            self.weights = np.load(self.weightfile)
            if self.has_bias:
                logger.info(f"Loading biases from {self.biasfile}")
                self.bias = np.load(self.biasfile) 
            return
               

    @staticmethod
    def layer_type(weights):
        """Given a framework layer, determine the weightwatcher LAYER_TYPE
        This can detect basic  PyTorch classes by type, and will try to infer the type otherwise. """

        the_type = LAYER_TYPE.UNKNOWN
        if len(weights.shape)==1:
            the_type = LAYER_TYPE.NORM
        elif len(weights.shape)==2:
            the_type = LAYER_TYPE.DENSE
        elif len(weights.shape)==4:
            the_type = LAYER_TYPE.CONV2D
        
        return the_type
    
    @staticmethod
    def layer_type_as_str(weights):
        """Given a framework layer, determine the weightwatcher LAYER_TYPE as a STRING
        This can detect basic  PyTorch classes by type, and will try to infer the type otherwise. """

        the_type = UNKNOWN
        if len(weights.shape)==1:
            the_type = NORM
        elif len(weights.shape)==2:
            the_type = DENSE
        elif len(weights.shape)==4:
            the_type = CONV2D
        
        return the_type
    
    
    
    def get_weights_and_biases(self):
        """   return has_weights, weights, has_biases, biases  """
        self.lazy_load_weights_and_biases()
        return self.has_weights, self.weights, self.has_bias, self.bias


    @staticmethod
    def get_layer_iterator(config, start_id=0):
        def layer_iter_():
            weights_dir =  config['weights_dir']
            logger.debug(f"iterating over layers in {weights_dir}")
            for layer_id, layer_config in config['layers'].items():
                layer_id = int(layer_id)+start_id
                py_layer = WWFlatFile(layer_id, config, layer_config)
                yield py_layer            
        return layer_iter_()   
    
        



class ONNXLayer(FrameworkLayer):
    """Helper class to support ONNX layers
    
    Turns out the op_type is option, so we have to 
    infers the layer_ type from the dimension of the weights 
        [a,b,c,d]  ->  CONV2D 
        [a,b]  ->  DENSE 
                
    Warning: this has not been tested in some time
    """
    
    def __init__(self, model, inode, node):
        
        self.model = model
        self.dims = node.dims     

        self.layer = node
        self.layer_id = inode
        self.name = node.name
        
       
        longname = self.name
            
        the_type = self.layer_type(self.dims)
        channels = CHANNELS.LAST
        FrameworkLayer.__init__(self, self.layer, self.layer_id, self.name, longname=longname, the_type=the_type, 
                                framework=FRAMEWORK.ONNX, channels=channels)     

    
    def set_weights(self, W):
        logger.fatal("Sorry set_weights not yet available for ONNX models")
         
    def get_weights(self):
        """ get the weights from the ONNX graph"""
        W= None
        
        idx = self.layer_id
        node_weights = [self.model.graph.initializer[idx]][0]
        
        if node_weights is not None:
            import onnx
            W = onnx.numpy_helper.to_array(node_weights)       
        
        return W

    def get_weights_and_biases(self):
        """extract the original weights (as a tensor) for the layer, and biases for the layer, if present
        
         Hacked together right now for issue #233
        """
        
        W =  self.get_weights()
        return True, W, False, None
    

    def layer_type(self, dims):
        """Given a framework layer, determine the weightwatcher LAYER_TYPE
        This can detect basic  PyTorch classes by type, and will try to infer the type otherwise. """
        
        the_type = LAYER_TYPE.UNKNOWN
        
        if len(self.dims) == 4:
            the_type = LAYER_TYPE.CONV2D
        elif len(self.dims) == 2:
            the_type = LAYER_TYPE.DENSE
        else:
            logger.debug("Unsupported ONNX Layer, dims = {}".format(self.dims))
                
        return the_type

        
            
    def replace_layer_weights(self, W, B=None):
        
        self.set_weights(W)
        if B is not None:
            logger.fatal("dont know hownto set Bias on ONNX models, stopping")


    @staticmethod
    def get_layer_iterator(model):
        def layer_iter_():
            for inode, node in enumerate(model.graph.initializer):
                yield ONNXLayer(model, inode, node)             
        return layer_iter_() 

    
        
        
class WWLayer:
    """WW wrapper layer to Keras and PyTorch Layer layer objects
       Uses python metaprogramming to add result columns for the final details dataframe"""
       
    def __init__(self, framework_layer, layer_id=-1, skipped=False, make_weights=True, params=None):
        
        if params is None: params = DEFAULT_PARAMS.copy()
        
        self.framework_layer = framework_layer
        self.layer_id = layer_id  
        self.skipped = skipped
        self.is_make_weights = make_weights
        self.params = params

        
        self.plot_id = framework_layer.plot_id
        self.name = framework_layer.name
        self.longname = framework_layer.longname


        
        self.the_type = framework_layer.the_type
        self.framework = framework_layer.framework      
        self.channels = framework_layer.channels
        
        self.fft = False

        # original weights (tensor) and biases
        self.has_weights = False
        self.weights = None
  
        self.has_biases = False
        self.biases = None
        
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
        data['longname'] = self.longname
        data['layer_type'] = str(self.the_type)
        data['N'] = self.N
        data['M'] = self.M
        data['rf'] = self.rf
        if self.M > 0:
            data['Q'] = self.N/self.M
        else:
            data['Q'] = -1
        
        for col in self.columns:
            data[col] = self.__dict__[col]
                    
        return data
    
    
    def __repr__(self):
        return "WWLayer()"

    def __str__(self):
        return "WWLayer {}  {} {}  skipped {}".format(self.layer_id, self.name,
                                                        self.the_type, self.skipped)

    def copy(self):
        """Create a shallow copy of WWLayer, sharing the framework_layer with the original."""
        # Create a new instance with the same properties
        copied_layer = WWLayer(
            framework_layer=self.framework_layer,  # Share the same framework_layer
            layer_id=self.layer_id,
            skipped=self.skipped,
            make_weights=self.is_make_weights,
            params=self.params.copy()  # Assuming params is a dictionary and needs a shallow copy
        )
        return copied_layer
    
    def __deepcopy__(self, memo):
        """Custom deep copy implementation for WWLayer."""
        # Create a new instance with copied properties
        copied_layer = WWLayer(
            framework_layer=self.framework_layer, 
            layer_id=self.layer_id,
            skipped=self.skipped,
            make_weights=self.is_make_weights,
            params=self.params
        )
        return copied_layer

    
    def make_weights(self):
        """ Constructor for WWLayer class.  Make a ww (wrapper)_layer from a framework layer, or return None if layer is skipped.
        In particular , late uses specify filter on layer ids and names """
        
        has_weights = False;
        if not self.skipped:
            has_weights, weights, has_biases, biases = self.framework_layer.get_weights_and_biases()
            
            self.has_weights = has_weights
            self.has_biases = has_biases
            
            if has_biases:
                self.biases = biases   
                self.has_biases = True
            
            if has_weights:    
                self.weights = weights
                self.set_weight_matrices(weights)
    
        return self
        

    #
    # TODO: maybe break this up into get dimensions and then set weights
    #
    def set_weight_matrices(self, weights, combine_weights_and_biases=False):#, conv2d_fft=False, conv2d_norm=True):
        """extract the weight matrices from the framework layer weights (tensors)
        sets the weights and detailed properties on the ww (wrapper) layer 
        
        Sets parameters like N, M, rf, Wmats, num_components, ...
    
        conv2d_fft not supported yet
        
        
        TODO: support W+b """
   
        if not self.has_weights:
            logger.info("Layer {} {} has no weights".format(self.layer_id, self.name))
            return 
        
        the_type = self.the_type
        conv2d_fft = self.params[CONV2D_FFT]
        
        N, M, n_comp, rf = 0, 0, 0, 1.0
        Wmats = []
        
        # this may change if we treat Conv1D layyer differently 
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
                Wmats, N, M, n_comp = self.get_conv2d_fft(weights)
            
        elif the_type == LAYER_TYPE.NORM:
            #logger.info("Layer id {}  Layer norm has no matrices".format(self.layer_id))
            pass
        
        #TODO: add Bibirectional Layer here
        
        elif isinstance(self.framework_layer, PyStateDictLayer):
            pass
        
        else:
            logger.info("Layer Type Unknown: id {}  unknown type {} layer  {}".format(self.layer_id, the_type, type(self.framework_layer)))
    
        self.N = N
        self.M = M
        self.rf = rf
        self.Wmats = Wmats
        self.num_components = n_comp
        
        if len(self.weights)>0:
            self.weight_dims = self.weights.shape
            self.num_params = np.prod(self.weight_dims)
        else:
            self.weight_dims = 0
            self.num_params = 0
            
        return 
        
        
    def get_conv2d_fft(self, W, n=32):
        """Compute FFT of Conv2D CHANNELS, to apply SVD later"""
        
        logger.info("get_conv2d_fft on W {}".format(W.shape))

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
            # flip how we extract the Wmats
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
        """unpermute the previously permuted, randomized weights"""
        
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
    
    
    def get_weights_and_biases(self):
        return self.framework_layer.get_weights_and_biases()
        
    def replace_layer_weights(self, W, B=None):
        return self.framework_layer.replace_layer_weights(W=W, B=B,)


    
    
class ModelIterator:
    """Iterator that loops over ww wrapper layers, with original matrices (tensors) and biases (optional) available.
    
    Note: only set the framework if the model has been already set
    
    """

    def __init__(self, model, framework = None, params=None, layer_map=None):
 
        if params is None: params = DEFAULT_PARAMS.copy()
        
        if model is None and framework is not None:
            logger.fatal("ModelIterator malformed")
            
        self.framework = framework
        if model is not None:
            self.model = model        
            self.framework = WeightWatcher.infer_framework(model)
            if WeightWatcher.valid_framework(self.framework):
                banner = WeightWatcher.load_framework_imports(self.framework)
                logger.info(banner)
                logger.info(f"framework from model = {self.framework}")
            else:
                logger.fatal("Could not infer framework from model, stopping")
                
         
        self.params = params
        if params[START_IDS]:
            start_id = params[START_IDS] 
        else:
            start_id = DEFAULT_START_ID
            if framework==FRAMEWORK.PYSTATEDICT:
                start_id = 1
                
        self.channels  = self.set_channels(params.get(CHANNELS_STR))
        
        self.layer_map = layer_map
        
        self.model_iter = self.model_iter_(model,start_id,layer_map)
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
        
    
    
    def model_iter_(self, model, start_id=0, layer_map=None):
        """Return a generator for iterating over the layers in the model.  
        Also detects the framework being used. 
        Used by base class and child classes to iterate over the framework layers 
        
        start_id = 0 is included for back compability; really all counting should start at 1
        
        layer_map used by subclasses for SafeTensors , PyStateDIct
        
        """
        layer_iter = None
        
     
        # sart_id can be erro (for back compatability) or one (better)
        if self.framework == FRAMEWORK.KERAS:
            layer_iter = KerasLayer.get_layer_iterator(model, start_id=start_id) 
            
                        
        elif self.framework == FRAMEWORK.KERAS_H5_FILE:
            model = WeightWatcher.read_keras_h5_file(model)
            self.model = model
            self.framework = FRAMEWORK.KERAS
            layer_iter = KerasLayer.get_layer_iterator(model, start_id=start_id) 

        elif self.framework == FRAMEWORK.PYTORCH:
            layer_iter = PyTorchLayer.get_layer_iterator(model,start_id=start_id) 
 
        elif self.framework == FRAMEWORK.ONNX:
            layer_iter = ONNXLayer.get_layer_iterator(model) 
            
        elif self.framework == FRAMEWORK.PYSTATEDICT:
            layer_iter = PyStateDictLayer.get_layer_iterator(model, start_id=start_id) 
            
        elif self.framework == FRAMEWORK.WW_FLATFILES:
            config = WeightWatcher.read_pystatedict_config(model)
            layer_iter = WWFlatFile.get_layer_iterator(config, start_id=start_id) 
             
        elif self.framework == FRAMEWORK.PYSTATEDICT_DIR:
            layer_iter = PyStateDictDir.get_layer_iterator(model, start_id=start_id, layer_map=layer_map)

            
        else:
            layer_iter = None
            
        return layer_iter
                      
    def make_layer_iter_(self):
        """The layer iterator for this class / instance.
         Override this method to change the type of iterator used by the child class"""
        return self.model_iter
    
    
    #TODO: this should be deprecated
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

    def __init__(self, model, framework, params=None, filters=[], layer_map=None):
        """ 
        This is the base LayerIterator, used for most calculations
        
        layer_map is only used by models where layers can be  accessed by name from a dict (i.e SafeTensors)
        
        
        """
        
        if params is None: params = DEFAULT_PARAMS.copy()
        super().__init__(model, framework=framework,  params=params, layer_map=layer_map)
                
        self.filters = filters
        
        self.filter_ids = []
        self.filter_types = []
        self.filter_names = []
        
        self.layer_map = layer_map
        
        #issue 312 in progress 
        #$if self.layer_map is not None and params[SAFETENSORS]==True:
        #    return
        
        if type(filters) is not list:
            filters = [filters]
            
        for f in filters:
            tf = type(f)

            if tf is str and f in LAYER_TYPES_SUPPORTED:
                logger.info("Filtering layer by type {}".format(str(f)))
                self.filter_types.append(f)
            elif isinstance(f, numbers.Integral):
                logger.info("Filtering layer by id {}".format(f))
                self.filter_ids.append(int(f)) 
            elif tf is str:
                logger.info("Filtering layer by name {}".format(f))
                self.filter_names.append(f) 
            else:
                logger.warning("unknown filter type {} detected and ignored".format(tf))
                
        return
        
                
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
            if np.min(self.filter_ids) >= 0:
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
            
            # can either remove 1 or names or keep 1 or names, not both
            bool_list = ["-" in x for x in self.filter_names]
            

            if not any(bool_list):
                if ww_layer.name in self.filter_names:
                    logger.debug("keeping layer {} {} by name ".format(ww_layer.layer_id, ww_layer.name))
                    ww_layer.skipped = False
                else:
                    logger.debug("skipping layer {} {} by name ".format(ww_layer.layer_id, ww_layer.name))
                    ww_layer.skipped = True
                
            elif all(bool_list):
                if f"-{ww_layer.name}" in self.filter_names:
                    logger.debug("skipping layer {} {} by name ".format(ww_layer.layer_id, ww_layer.name))
                    ww_layer.skipped =True
                else:
                    logger.debug("keeping layer {} {} by name ".format(ww_layer.layer_id, ww_layer.name))
                    ww_layer.skipped = False
                    
            else:
                logger.critical("Input error, can only specify a list of names to keep or a lsit to skip")
     
        return ww_layer.skipped
    
    
    
    def ww_layer_iter_(self):
        """Create a generator for iterating over ww_layers"""
        
        for curr_layer in self.model_iter:
            
            is_skipped = True
            is_supported = False
            
            if curr_layer.lazy and not curr_layer.skipped:
                is_skipped = self.apply_filters(curr_layer)
                if not is_skipped:
                    is_supported = self.layer_supported(curr_layer)
                if not is_skipped and is_supported:
                    logger.debug(f"creating lazy layer {curr_layer.layer_id}")
                    ww_layer = WWLayer(curr_layer, layer_id=curr_layer.layer_id, params=self.params)
                    ww_layer.skipped = False
                else:
                    logger.debug(f"skipping lazy layer {curr_layer.layer_id}")
            else:     
                ww_layer = WWLayer(curr_layer, layer_id=curr_layer.layer_id, params=self.params)
                is_skipped = self.apply_filters(ww_layer)
                is_supported = self.layer_supported(ww_layer)

            if is_supported and not is_skipped:
                yield ww_layer    
                
                
                
    def make_layer_iter_(self):
        return self.ww_layer_iter_()
    
    
  
    def make_layer_map(self):
        """ Make a copy of the iterator, iterate over all layers, and make the layer_map
           
            We use cls=type(self) to allow recreating the iterator from a subclass
        """

        if self.layer_map is None:
            cls = type(self)
            iter_copy = cls(self.model, self.framework, params=self.params, filters=self.filters, layer_map=None)
            self.layer_map = [ww_layer.name for ww_layer in iter_copy.make_layer_iter_()]
    
        return self.layer_map
    
    
    
    def layer_supported(self, ww_layer):
        """Return true if this kind of layer is supported
        
        Supports the WWLayer, or a Lazy FranmeWork layer
        
        Layer object must have the following fileds:
            layer_id, name, longname, the_type, N, M, rf
        """
        
        supported = False

        layer_id = ww_layer.layer_id
        name = ww_layer.name
        longname = ww_layer.longname
        the_type = ww_layer.the_type
        
        N = ww_layer.N
        M = ww_layer.M
        rf = ww_layer.rf
        
        min_evals = self.params.get(MIN_EVALS)
        max_evals = self.params.get(MAX_EVALS)
        max_N = self.params.get(MAX_N)

        pool = self.params.get(POOL)
        # experimental
        if self.params.get(FFT):
            rf = 1.0
            
        # deprecated
        # ww2x = self.params.get(WW2X)
        conv2d_fft = self.params.get(CONV2D_FFT)
        
        logger.debug("layer_supported  N {} max evals {}".format(N, max_evals))
        
        if ww_layer.skipped:
            logger.debug("Layer {} {} is skipped".format(layer_id, name))
            return False
            
        elif not ww_layer.has_weights:
            logger.debug("layer not supported: Layer {} {} has no weights".format(layer_id, name))
            return False
        
        elif the_type is LAYER_TYPE.UNKNOWN:
            logger.debug("layer not supported: Layer {} {} type {} unknown".format(layer_id, name, the_type))
            return False
        
        elif the_type in [LAYER_TYPE.FLATTENED, LAYER_TYPE.NORM]:
            logger.debug("layer not supported: Layer {} {} type {} not supported".format(layer_id, name, the_type))
            return False
        
        
        elif not pool and min_evals and M  <  min_evals:
            logger.debug("layer not supported: Layer {} {}: num_evals {} <  min_evals {}".format(layer_id, name, M, min_evals))
            return False
                  
        elif not pool and max_evals and M  >  max_evals:
            logger.debug("layer not supported: Layer {} {}: num_evals {} > max_evals {}".format(layer_id, name, N, max_evals))
            return False

        elif (pool) and (not conv2d_fft) and min_evals and M * rf < min_evals:
            logger.debug("layer not supported: Layer {} {}: num_evals {} <  min_evals {}".format(layer_id, name, M * rf, min_evals))
            return False
                  
        elif (pool)  and (not conv2d_fft) and max_evals and M * rf > max_evals:
            logger.debug("layer not supported: Layer {} {}: num_evals {} > max_evals {}".format(layer_id, name,  M * rf, max_evals))
            return False
        
        elif N > max_N:
            logger.debug("layer not supported: Layer {} {}: N*rf {} > max_evals {}".format(layer_id, name, N , max_N))
            return False
        
        elif the_type in [LAYER_TYPE.DENSE, LAYER_TYPE.CONV1D, LAYER_TYPE.CONV2D, LAYER_TYPE.EMBEDDING]:
            supported = True
            
               
        return supported
    

class WW2xSliceIterator(WWLayerIterator):
    """Iterator variant that breaks Conv2D layers into slices for back compatability; used when NOT POOLING"""
    from copy import deepcopy

    def ww_slice_iter_(self):
        
        for ww_layer in self.ww_layer_iter_():
            if ww_layer.the_type == LAYER_TYPE.CONV2D:
                layer_id = ww_layer.layer_id                
                count = len(ww_layer.Wmats)
                for iw, W in enumerate(ww_layer.Wmats):
                    ww_slice = deepcopy(ww_layer)
                    ww_slice.Wmats = [W]
                    ww_slice.conv2d_count = count

                    slice_id = iw
                    ww_slice.add_column("slice_id", slice_id)
                    ww_slice.plot_id = f"{layer_id}:{slice_id}"

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
            
            i.e:  W0.shape[1]==W1.shape[0] along the shared dimension  
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
                    logger.warning("Cannot align W0={} with W1={}".format(W0.shape, W1.shape))
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
    - Only supports pool=True (ww2x=False )
    
    - The layer can be analyzed, but does not yet support SVDSmoothing, etc 
    
    - Each layer matrix is normalized by the Frobenius norm  W=W/||W||_F

    - Each layer matrix is padded with zeros to the right,  i.e
    
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
            
            # Here, I just lazily copy an older layer
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
            W = (W - np.median(W))/stats.median_abs_deviation(W)
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
        
        
        # This needs to be reset and used properly , eventually
        # the effective M, used for Q, will be much smaller 
        # if there are a huge number of zero eigenvalues 
        self.num_components = M  # default for full SVD, not used yet
        
        yield ww_stacked_layer
                
    def make_layer_iter_(self):
        return self.ww_stacked_iter_()
    


    
class WWPeftLayerIterator(WWLayerIterator):
    """ Iterates over a layer and creates an extended model
      
        For every layer, we check it there is a LORA update
        If present,  
           the A layer is replaced with full lora update,  np.dot(A.T, B.T)
           the B layer is W + Lora update
           
        This will NOT be called  if params.PEFT=False
        
        Assumes base_layer, lora_a, lora_b are in this order
    
        # TODO:  check names ?
        
        PEFT_WITH_BASE => compute the base layer if it can be found
             (not really supported, legacy code)
    """
    
    def __init__(self, model, framework, params=None, filters=[]):
        self.peft_only = params[PEFT]!=PEFT_WITH_BASE
        super().__init__(model, framework=framework, filters=filters, params=params)   
    
    
    def update_peft_layers(self, A_layer, B_layer, last_W):
        transpose = False
        A = A_layer.Wmats[0]
        B = B_layer.Wmats[0]


        # switch order or not ?
        if A.shape[0] > A.shape[1]:
            logger.debug("Transposing A ")
            transpose = True
            A = A.T

        if B.shape[1]!=A.shape[0]:
            logger.debug("Transposing B ")
            B = B.T

        if B.shape[1]!=A.shape[0]: 
            logger.warning(f"LoRA matrices can not be aligned AB shape as {B.shape} x {A.shape} ")

        lora_rank = min(A.shape)

        BA = np.dot(B,A)  

        A_layer.Wmats[0] = BA
        A_layer.N = max(BA.shape)
        A_layer.M = min(BA.shape)
        A_layer.name = A_layer.name.replace("lora_A", "lora_BA")   
        A_layer.longname = A_layer.longname.replace("lora_A", "lora_BA")    
        A_layer.num_components = lora_rank
        A_layer.peft = True

        # if we are onlyb analyzing the adpater_model.bin. this is skipped
        # not entirely sure about the updates with transpose
        if last_W is None:  
            transpose = False

        else:
            if transpose:  
                updated_W = last_W + BA.T
            else:
                updated_W = last_W + BA


            B_layer.Wmats[0] = updated_W
            B_layer.N = max(updated_W.shape)
            B_layer.M = min(updated_W.shape)
            B_layer.name = B_layer.name.replace("lora_B", "lora_W_plus_AB")
            B_layer.longname = B_layer.longname.replace("lora_B", "lora_W_plus_AB")
            B_layer.num_components = lora_rank

            if self.peft_only:
                B_layer.skipped = True
            else:
                B_layer.peft = True
                
        return B_layer
                
                                
                
    
    def ww_peft_iter(self):
    
        last_W = None
        AB_layers = []
               
        for ww_layer in self.ww_layer_iter_():
            ww_layer.add_column("peft", False)
            if ww_layer.the_type == LAYER_TYPE.DENSE and 'lora_' in ww_layer.longname:
            
                if 'lora_A' in ww_layer.longname:
                    
                    A_layer = ww_layer  
                    A = A_layer.Wmats[0]
                    AB_layers.append(A_layer)
                        
                elif 'lora_B' in ww_layer.longname:
                    B_layer = ww_layer
                    
                    if self.peft_only:
                        last_W = None
                        
                    self.update_peft_layers(A_layer, B_layer, last_W)
                    
                    if last_W is not None:  
                        if self.peft_only:
                            B_layer.skipped = True
                        else:
                            B_layer.peft = True
                        AB_layers.append(B_layer)

                    while len(AB_layers)>0:
                        yield AB_layers.pop()
            
                else:
                    logger.warning(f"unknown lora layer {ww_layer.longname}")
            
            else:
                last_W = ww_layer.Wmats[0]
            
                if self.peft_only:
                    ww_layer.skipped = True
                yield ww_layer
    
                
                
    def make_layer_iter_(self):
        return self.ww_peft_iter()
    
    


class WWDeltaLayerIterator(WWLayerIterator):
    """ Combines 2 layer iterators, and iterates over 2 layers
    
        W_delta := W_model - W_base  (biases currently ignored)
        
        TODO:  ensure layers are aligned (when using 1 safetensors, 1 other)
        ALSO:  need layer_id map for correlation flow
        
        ALSO:  can we 'force' the framework ?
        
        NOTE: filters don't work here with safetensors:  see  git issue #312
    
    """
    
    def __init__(self, base_model, base_framework,  model, framework, filters=[], params=None, Iterator_Class=WWLayerIterator):
    
        if params is None: params = DEFAULT_PARAMS.copy()
        
        self.iter_left = Iterator_Class(model, framework=framework,   filters=filters, params=params)   
        
        # warning: this creates a copy of the iterator and iterates over the all layers
        layer_map = self.iter_left.make_layer_map()
        if layer_map is not None and len(layer_map)>0:
            logger.info(f"Using model layer map, num layers={len(layer_map)}")
            
        # warning: layer_map may be ignored
        self.iter_right = Iterator_Class(base_model, framework=base_framework,  filters=filters, params=params, layer_map=layer_map)   
        
        super().__init__(model, framework=framework, filters=filters, params=params, layer_map=layer_map)   

    #
    # # TODO: deal with case when base does not contain a layer
    # # layers may be added wnen FTing
    # def ww_layer_iter_(self):   
    #     from copy import deepcopy
    #
    #     for left_layer, right_layer in zip(self.iter_left, self.iter_right):
    #
    #             logger.info(f"{left_layer.name} <-> {right_layer.name}")
    #             base_name = left_layer.name
    #             model_name = right_layer.name
    #
    #             # TODO: check longnames first
    #             # 
    #
    #             # sometimes we need to skip the first (and last) layer 
    #             if base_name not in model_name:
    #                 logger.warning(f"Skipping base model layer {base_name}")
    #
    #                 left_layer = self.iter_left.next()
    #                 base_name = left_layer.name
    #                 logger.info(f"{left_layer.name} <-> {right_layer.name}")       
    #
    #             ww_layer = deepcopy(left_layer)
    #
    #             if ww_layer.has_weights:
    #                 left_Wmats = left_layer.Wmats
    #                 right_Wmats = right_layer.Wmats
    #                 if len(right_Wmats) > 0:   
    #                     Wmats = []
    #                     for W_left,  W_right in zip(left_Wmats, right_Wmats):
    #                         if W_left is not None:
    #                             if W_left.shape == W_right.shape:
    #                                 Wmats.append(W_left -  W_right )
    #                             elif W_left.shape == W_right.T.shape:
    #                                  Wmats.append(W_left -  W_right.T )
    #                             else: 
    #                                 logger.warning(f"W_left.shape, W_right.shape not compatable {W_left.shape } != {W_right.shape }")
    #                                 W_zero = np.zeros_like(W_left)
    #                                 Wmats.append(W_zero)
    #                         else:
    #                             Wmats.append(None)
    #                     ww_layer.Wmats = Wmats 
    #                 else:
    #                     logger.warning(f"skiping base model layer left_layer. name, not removing")
    #
    #             #  its possible the fine tuned model did not update the biases...may need to treat this case too
    #             if ww_layer.has_biases:
    #                 left_biases = left_layer.biases
    #                 right_biases = right_layer.biases
    #                 biases = []
    #                 for B_left,  B_right in zip(left_biases, right_biases):
    #                     if B_left is not None:
    #                         biases.append(B_left -  B_right )
    #                     else:
    #                         biases.append(None)
    #                 ww_layer.biases = biases 
    #
    #             yield ww_layer    
    #
    #
    #
    #     return self.ww_layer_iter_()
    
    def ww_layer_iter_(self):
        """
        Iterate over layers from two models (`iter_left` and `iter_right`), processing them in pairs when possible. 
        If a layer from `iter_left` does not have a corresponding layer in `iter_right`, it is processed independently.
        
        Yields:
            ww_layer: A processed layer from the `iter_left` model with any differences computed against the corresponding
                      `iter_right` model layer if available.
        """
    
        from copy import deepcopy
    
        # Create an iterator for the right layers
        right_iter = iter(self.iter_right)
        right_layer = next(right_iter, None)
            
        for left_layer in self.iter_left:
            base_name = left_layer.name
            logger.info(f"processing {base_name}")

            # Check if there is a corresponding right layer
            if right_layer is not None and base_name in right_layer.name:
                # Matching layer found
                logger.info(f"{left_layer.name} <-> {right_layer.name}")
                model_name = right_layer.name
    
                ww_layer = deepcopy(left_layer)
    
                if ww_layer.has_weights:
                    left_Wmats = left_layer.Wmats
                    right_Wmats = right_layer.Wmats
                    Wmats = []
                    for W_left, W_right in zip(left_Wmats, right_Wmats):
                        if W_left is not None:
                            if W_left.shape == W_right.shape:
                                Wmats.append(W_left - W_right)
                            elif W_left.shape == W_right.T.shape:
                                Wmats.append(W_left - W_right.T)
                            else:
                                logger.warning(f"W_left.shape, W_right.shape not compatible {W_left.shape} != {W_right.shape}")
                                W_zero = np.zeros_like(W_left)
                                Wmats.append(W_zero)
                        else:
                            Wmats.append(None)
                    ww_layer.Wmats = Wmats
    
                # Handle biases
                if ww_layer.has_biases:
                    left_biases = left_layer.biases
                    right_biases = right_layer.biases
                    biases = []
                    for B_left, B_right in zip(left_biases, right_biases):
                        if B_left is not None:
                            biases.append(B_left - B_right)
                        else:
                            biases.append(None)
                    ww_layer.biases = biases
    
                # Move to the next right layer
                right_layer = next(right_iter, None)
    
            else:
                # No matching right layer, process the left layer independently
                logger.info(f"Processing left layer {left_layer.name} independently")
                ww_layer = deepcopy(left_layer)
    
                if ww_layer.has_weights:
                    ww_layer.Wmats = left_layer.Wmats
    
                if ww_layer.has_biases:
                    ww_layer.biases = left_layer.biases
    
            # Yield the processed layer
            yield ww_layer
    
        return self.ww_layer_iter_()


    
class WeightWatcher:

    # add base model
    def __init__(self, model=None, framework=None, base_model=None, log_level=None):
        """ model is set or is none
            the framework can be set or it is inferred
        
            valid frameworks = 'keras' | 'pytorch' | 'onnx' | ww.KERAS | ww.PYTORCH | ww.ONNX
        
            log_level can be set may not currently work """
        
        logger.info(f" __init__: {model}, {base_model}")

        
        if log_level:
            logger.setLevel(log_level)
        
        self.model = model
        self.base_model = base_model
        
        self.details = None
        self.framework = None
        
        self.results = None

        banner = self.banner()

        if model is not None:
            framework = self.infer_framework(model)
            if WeightWatcher.valid_framework(framework):
                self.framework = framework
                banner += "\n"+ self.load_framework_imports(framework)
            else:
                logger.fatal("Could not infer framework from model, stopping")
                
        if model is None and base_model is not None:
            logger.fatal("Model can not be none if base_model is not None")
            
        if base_model is not None:
            base_framework = self.infer_framework(base_model)
            if not WeightWatcher.valid_framework(base_framework):
                logger.fatal("Could not infer framework from base_model, stopping")
            elif base_framework != self.framework:
                banner += "\n"+ self.load_framework_imports(base_framework)
        
            if base_framework!=self.framework:
                 logger.fatal(f"base_framework {base_framework} is different from model framework {self.framework}, stopping")
                 
        logger.info(banner)
        
        
  
    
    # TODO: fix or deprecate
    def header(self):
        """WeightWatcher v0.1.dev0 by Calculation Consulting"""
#        from weightwatcher import __name__, __version__, __author__, __description__, __url__
#        return "{} v{} by {}\n{}\n{}".format(__name__, __version__, __author__, __description__, __url__)
        return ""

    # TODO: improve banner
    def banner(self):
        versions = "\npython      version {}".format(sys.version)
        versions += "\nnumpy       version {}".format(np.__version__)            
        #versions += "\ntensforflow version {}".format(tf_version)
        #versions += "\nkeras       version {}".format(keras_version)
        return "\n{}{}".format(self.header(), versions)

    def __repr__(self):
        done = bool(self.results)
        txt = "\nAnalysis done: {}".format(done)
        return "{}{}".format(self.header(), txt)
   
        
    @staticmethod
    def valid_framework(framework):
        """is a valid FRAMEWORK constant """
        valid = framework in [ FRAMEWORK.KERAS, FRAMEWORK.PYTORCH, FRAMEWORK.PYSTATEDICT, FRAMEWORK.PYSTATEDICT_DIR, FRAMEWORK.ONNX,  FRAMEWORK.WW_FLATFILES]
        return valid
        
    
    @staticmethod
    def infer_framework(model):
        """Tries to infer the model framework.
        If a directory, looks inside to find model formats,  and in this order:  
                 ECTORY  (SAFETENSORS, PYTORCH), WWFlatFile (i.e. numpy flat file))
        
        
        """
          
        def is_framework(name='UNKNOWN'):
            found = False
            for cls in inspect.getmro(type(model)):
                found |= name in str(cls)
                                                          
            return found 
    
        framework = FRAMEWORK.UNKNOWN
        if model is not None:
            
            if is_framework(name='torch'):
                return FRAMEWORK.PYTORCH
            elif is_framework(name='keras'):
                return FRAMEWORK.KERAS 
            elif is_framework(name='onnx'):
                return FRAMEWORK.ONNX
            elif is_framework(name='OrderedDict') or is_framework(name='dict'):
                # currently only pystatedict is supported
                # but this could be changed
                # we could dig inside the model and find the weight types, 
                return FRAMEWORK.PYSTATEDICT
            
            elif os.path.isdir(model):
                format, fileglob = WeightWatcher.infer_model_file_format(model)
                if format in [MODEL_FILE_FORMATS.PYTORCH, MODEL_FILE_FORMATS.SAFETENSORS]:
                    return FRAMEWORK.PYSTATEDICT_DIR
                elif format == MODEL_FILE_FORMATS.WW_FLATFILES:
                    return FRAMEWORK.WW_FLATFILES
                elif format == FRAMEWORK.KERAS_H5_FILE:
                    return FRAMEWORK.KERAS_H5_FILE
                else:
                    logger.fatal(f"Directory {model} does not contain any known files")
            #
            # elif model is a json fole
            #    return FRAMEWORK.KERASJSONFILE
                
        return framework


    @staticmethod
    def infer_model_file_format(model_dir):
        """inspect the files in the model_dir, determine the fileformat
        
            looks first for SAFETENSORS
            if not found, looks for PYTORCH
            
            format = MODEL_FILE_FORMATS.SAFETENSORS | MODEL_FILE_FORMATS.PYTORCH | WW_FILE | UNKNOWN
            
            returns: format, fileglob
        """
             
        format = UNKNOWN
        fileglob = UNKNOWN
                
        # first, try SAFETENSORS
        fileglob = f"{model_dir}/model*safetensors"
        num_files = len(glob.glob(fileglob))
        if num_files > 0:
            format = MODEL_FILE_FORMATS.SAFETENSORS
            return format, fileglob
        
        # first, try SAFETENSORS
        fileglob = f"{model_dir}/adapter_model*safetensors"
        num_files = len(glob.glob(fileglob))
        if num_files > 0:
            format = MODEL_FILE_FORMATS.SAFETENSORS
            return format, fileglob
       
       
        fileglob = f"{model_dir}/pytorch_model*bin"
        num_files = len(glob.glob(fileglob))
        if num_files > 0:
            format = MODEL_FILE_FORMATS.PYTORCH
            return format, fileglob
        
        
        fileglob = f"{model_dir}/adapter_model*bin"
        num_files = len(glob.glob(fileglob))
        if num_files > 0:
            format = MODEL_FILE_FORMATS.PYTORCH
            return format, fileglob

        
        fileglob = f"{model_dir}/*weight*npy"
        num_files = len(glob.glob(fileglob))
        if num_files > 0:
            format = MODEL_FILE_FORMATS.WW_FLATFILES
            return format, fileglob
        
        
        fileglob = f"{model_dir}/*h5"
        num_files = len(glob.glob(fileglob))
        if num_files > 0:
            format = MODEL_FILE_FORMATS.KERAS_H5_FILE
            return format, fileglob

                 
        format = UNKNOWN
        fileglob = UNKNOWN
            
        return format, fileglob

   
    @staticmethod
    def load_framework_imports(framework):
        """load tensorflow, pytorch, or onnx depending on the framework 
        
        returns a banner to display """

        banner = ""
        if framework==FRAMEWORK.KERAS:
            #import tensorflow as tf
            #from tensorflow import keras
            
            global tf, keras
            try:
                tf = importlib.import_module('tensorflow')
                keras = importlib.import_module('tensorflow.keras')
        
                banner = f"tensorflow version {tf.__version__}"+"\n"
                #banner += f"keras version {keras.__version__}"
            except ImportError:
                logger.fatal("Can not load tensorflow or keras, stopping")
            
        elif framework in [FRAMEWORK.PYTORCH, FRAMEWORK.PYSTATEDICT, FRAMEWORK.PYSTATEDICT_DIR, FRAMEWORK.WW_FLATFILES] :
            
            global torch, nn
            try:
                torch = importlib.import_module('torch')
    
                banner = f"torch version {torch.__version__}"
            except ImportError:
                logger.fatal("Can not load torch,  stopping")
            

        #TODO: this doesn't work; IDK why yet, hacked in the ONNX layer
        elif framework==FRAMEWORK.ONNX:
            try:
                import onnx
                from onnx import numpy_helper
                banner = f"onnx version {onnx.__version__}"   
            except ImportError:
                logger.fatal("Can not load ONNX, stopping")
                
        else:
            logger.warning(f"Unknown or unsupported framework {framework}")
            banner = ""
                
        return banner
    
    
    # # TODO: moved from iterator    
    # # redo such that we don't have to include modules until framework detected
    # @deprecated
    # def set_framework(self, model, framework=None):
    #     """Sets the framework (if specified) or infers it
    #
    #      """
    #
    #     framework = FRAMEWORK.UNKNOWN
    #     if hasattr(self.model, LAYERS):
    #         framework = FRAMEWORK.KERAS
    #
    #     elif hasattr(self.model, 'modules'):
    #         framework = FRAMEWORK.PYTORCH
    #
    #     elif isinstance(self.model, onnx.onnx_ml_pb2.ModelProto):  #@pydevd suppress warning
    #
    #         framework = FRAMEWORK.ONNX
    #
    #     elif isinstance(self.model, str):
    #         if os.path.exists(self.model) and os.path.isdir(self.model):  
    #             logger.info("Expecting model is a directory containing pyTorch state_dict files")
    #             framework = FRAMEWORK.PYSTATEDICT
    #         else:
    #             logger.error(f"unknown model folder {self.model}")
    #
    #     return framework
    
    
    def same_models(self, model_1, model_2):
        """Compare models to see if they are the same architecture.
        Not really impelemnted yet"""
    
        same = True
        
        framework_1 = self.infer_framework(model_1)
        framework_2 = self.infer_framework(model_2)

        same = (framework_1 == framework_2)
        if not same:
            return False
        
        layer_iter_1 = WWLayerIterator(model_1)
        layer_iter_2 = WWLayerIterator(model_2)
        #TODO: finish     

        return same


    # TODO: unit test (and move to RMT util?)
    def matrix_distance(self, W1, W2, method=EUCLIDEAN):
        """helper method to compute the matrix distance or overlap
        
         currently supports EUCLIDEAN, CKA (uncentered) 
         
         Note: this works for 1-d matrices (vectors) also
         
        """
        
        dist = 0.0
        # valid method ?
        valid_params = method in [RAW, EUCLIDEAN, CKA]

        valid_input = True
        if W1 is None or W2 is None:
            logger.warning("Weight matries are Null")
            valid_input = False
        elif W1.shape!=W2.shape:
            logger.warning(f"Weight matrices are different shapes:  {W1.shape} =/= {W2.shape}")
            valid_input = False

        if not valid_params or not valid_input:
            logger.fatal("invalid input, stopping")
            return ERROR


        if method in [RAW, EUCLIDEAN]:
            dist = np.linalg.norm(W1-W2)
        elif method==CKA:
            # TODO:  replace with a call to the Apache 2.0 python codde for CKA
            # These methods will be add to RMT_Util or just from CKA.oy directly
            
            dist = np.linalg.norm(np.dot(W1.T,W2))
            norm1 =  np.linalg.norm(np.dot(W1.T,W1))
            norm2 =  np.linalg.norm(np.dot(W2,W2.T))
            norm = np.sqrt(norm1*norm2)
            if norm < 0.000001:
                norm = norm + 0.000001
            dist = dist / norm
        else:
            logger.warning(f"Unknown distances method {CKA}")

        return dist


    def model_deltas(self, model_1, model_2, layers = [], start_ids = 0):
        """Compute the deltas  (or difference) between the weight matrices of two models"""
        
    
    # TODO:
    # replace with iterator that forms T2-T2
    # then, raw is just T-T 4D tensor
    # and not raw is just pool=False
    #
   
    
    def distances(self, model_1, model_2,  method = RAW,
                  layers = [], start_ids = 0, pool = True, channels = None):
        """Compute the distances between model_1 and model_2 for each layer. 
        Reports Frobenius norm of the distance between each layer weights (tensor)
        

        methods: 
             'raw'      ||W_1-W_2|| , but using raw tensores (not supported yet)

             'euclidean'      ||W_1-W_2|| , using layer weight matrices that are extracted

             'cka'     || W_1 . W_2|| / ||W1|| ||W12||   (not centered yet)
           
        output: avg delta W, a details dataframe
           
        models should be the same size and from the same framework
           
           
        Note: conv2d_ftt is not supported
        """
        
        params = DEFAULT_PARAMS.copy()
        # not implemented here : 
        #params[CONV2D_FFT] = conv2d_fft
        
        params[POOL] = pool  
        params[CHANNELS_STR] = channels
        params[LAYERS] = layers
        # not implemented here:
        # params[STACKED] = stacked
        params[START_IDS] = start_ids

        logger.info("params {}".format(params))
        if not WeightWatcher.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)
        
        #if method==CKA and ww2x is not True:
        #    msg = "can not process Conv2D layers with CKA unless ww2x=True | pool=False"
        #    logger.error(msg)
        #    raise Exception(msg)

        #  specific distance input checks here
        if method is None:
            method == RAW
        else:
            method = method.lower()
        if method not in [RAW, EUCLIDEAN, CKA]:
            msg = "Error, method not valid: \n {}".format(method)
            logger.error(msg)
            raise Exception(msg)
            

        same = True
        layer_iter_1 = self.make_layer_iterator(model=model_1, layers=layers, params=params)           
        layer_iter_2 = self.make_layer_iterator(model=model_2, layers=layers, params=params)           
        
        same = layer_iter_1.framework == layer_iter_2.framework 
        if not same:
            raise Exception("Sorry, models are from different frameworks")
        
        distances = pd.DataFrame()
        data = {}
        ilayer = 0
        try:      
            for layer_1, layer_2 in zip(layer_iter_1, layer_iter_2):
                data['layer_id'] = layer_1.layer_id
                if hasattr(layer_1, 'slice_id'):
                    data['slice_id'] = layer_1.slice_id
                data['name'] = layer_1.name
                data['longname'] = layer_1.longname
                data['method'] = method

                if method in [RAW]:
                    if layer_1.has_weights:
                        has_weights1, W1, has_biases1, b1  = layer_1.get_weights_and_biases()
                        has_weights2, W2, has_biases2, b2  = layer_2.get_weights_and_biases()
                        
                        data['M'] = np.min(W1.shape)
                        data['N'] = np.max(W1.shape)
                        data['delta_W'] = self.matrix_distance(W1, W2, method)

                        if b1 is not None and b2 is not None and len(b1)==len(b2):       
                            data['b_shape'] = b1.shape
                            data['delta_b'] = self.matrix_distance(b1, b2, method)
                        else:
                            data['b_shape'] = UNKNOWN
                            data['delta_b'] = 0
                            
                # THiS should work for ww2x=True|pool=False. but not tested yet
                elif method in [EUCLIDEAN, CKA]:
                    if layer_1.has_weights:
                        W1_mats = layer_1.Wmats
                        W2_mats = layer_2.Wmats
                        
                        data['M'] = np.min(W1_mats[0].shape)
                        data['N'] = np.max(W1_mats[0].shape)
                        data['delta_W'] = 0.0
    
                        data['b_shape'] = UNKNOWN
                        data['delta_b'] = 0.0
                                
                        for W1, W2 in zip(W1_mats,W2_mats):
                            data['delta_W'] += self.matrix_distance(W1, W2, method)
                        data['delta_W'] /= float(len(W1_mats))
                                           
                else:
                    logger.fatal(f"unsupported distance method {method}")

                data_df = pd.DataFrame.from_records(data, index=[ilayer])
                distances = pd.concat([distances, data_df], ignore_index=True)
                ilayer += 1

        except:
            msg = "Oops!"+ str(sys.exc_info()[0])+ "occurred."
            logger.error("Sorry, problem comparing models")
            logger.error(msg)
            raise Exception("Sorry, problem comparing models: "+msg)

        # Reorder the columns so that layer_id and name come first.
        lead_cols = ['method', 'layer_id', 'name', 'delta_W', 'delta_b', 'M', 'N', 'b_shape']
        distances = distances[lead_cols + [c for c in distances.columns if not c in lead_cols]]

        distances.set_index('layer_id', inplace=True)
        avg_dW = np.mean(distances['delta_W'].to_numpy())
        avg_db = np.mean(distances['delta_b'].to_numpy())

        return avg_dW, avg_db, distances
    
    
    
    def combined_eigenvalues(self, Wmats, N, M, n_comp, params=None):
        """Compute the eigenvalues for all weights of the NxM weight matrices (N >= M), 
            combined into a single, sorted, numpy array
    
            Applied normalization and glorot_fix if specified
    
            Assumes an array of weights comes from a conv2D layer and applies conv2d_norm normalization by default
    
            Also returns max singular value and rank_loss, needed for other calculations
         """
    
        if params is None: params = DEFAULT_PARAMS.copy()

        all_evals = []
        max_sv = 0.0
        min_sv = 0.0

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

            W = W.astype(float)
            logger.debug("Running {} SVD:  W.shape={}  n_comp = {}".format(params[SVD_METHOD], W.shape, n_comp))
            
            # if peft, need to use Truncated SVD
            if (params[PEFT] and n_comp < M) or (n_comp < M):
                sv = svd_values(W, k=n_comp)
                sv = sv.flatten()
                sv = np.sort(sv)
            else:
                sv = svd_vals(W, method=params[SVD_METHOD])
                sv = sv.flatten()
                sv = np.sort(sv)[-n_comp:]
                # TODO:  move to PL fit for robust estimator
                # if len(sv) > max_evals:
                #    #logger.info("chosing {} singular values from {} ".format(max_evals, len(sv)))
                #    sv = np.random.choice(sv, size=max_evals)
    
            if params[INTRA]:
                evals = sv
                sv  = np.sqrt(sv)
            else:
                evals = sv * sv
            
            #if normalize:
            #    evals = evals / N
    
            #all_evals.extend(evals)
            # hacked this to fix bug
            all_evals = np.concatenate((np.array(all_evals), np.array(evals)))

            
            if params[INVERSE]:
               all_evals = 1.0/np.array(all_evals)
            else:
               all_evals = np.array(all_evals)
    
            max_sv = np.max([max_sv, np.max(sv)])
            min_sv = np.max([min_sv, np.min(sv)])

            rank_loss = rank_loss + calc_rank_loss(sv, N)      
    
        return np.sort(all_evals), max_sv, min_sv, rank_loss
            
            
    def apply_normalize_Wmats(self, ww_layer, params=None):
        """Normalize the W matrix or Wmats """

        if params is None: params = DEFAULT_PARAMS.copy()
        
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
                
        
        
    def apply_FFT(self, ww_layer, params=None):
        """compute the 2D fft of the layer weights, take real space part (probably not so useful)"""
                
        layer_id = ww_layer.layer_id
        name = ww_layer.name
       
        if not ww_layer.skippe:
            logger.info("applying 2D FFT on to {} {} ".format(layer_id, name))
            
            Wmats = ww_layer.Wmats
            for iw, W in enumerate(Wmats):
                W = np.real(np.fft.fft2(W))
                ww_layer.Wmats[iw]=W
                
            ww_layer.fft = True
        else:
            logger.info("skipping 2D FFT  for  Layer {} {} ".format(layer_id, name))

            
        return ww_layer 


    
                 
    def apply_esd(self, ww_layer, params=None):
        """run full SVD on layer weight matrices, compute ESD on combined eigenvalues, combine all, and save to layer """
        
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
                                
        evals, sv_max, sv_min, rank_loss = self.combined_eigenvalues(Wmats, N, M, n_comp, params)
        
        if params[TOLERANCE]:
            tolerance = params[TOLERANCE]
        else:
            tolerance = WEAK_RANK_LOSS_TOLERANCE
        weak_rank_loss = len(evals[evals<tolerance])
     
        ww_layer.evals = evals
        ww_layer.add_column("has_esd", True)
        ww_layer.add_column("num_evals", len(evals))
        ww_layer.add_column("sv_max", sv_max)
        ww_layer.add_column("sv_min", sv_min)
        ww_layer.add_column("rank_loss", rank_loss)
        ww_layer.add_column("weak_rank_loss", weak_rank_loss)
        ww_layer.add_column("lambda_max", np.max(evals))

            
        return ww_layer
    
    def apply_random_esd(self, ww_layer, params=None):
        """Randomize the layer weight matrices, compute ESD on combined eigenvalues, combine all,  and save to layer """
        
        if params is None: params = DEFAULT_PARAMS.copy()
        
        layer_id = ww_layer.layer_id
        plot_id =  ww_layer.plot_id
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
            
            # should be very close to bulk_max / lambda_max
            value = np.max(rand_evals)/np.max(evals)
            ww_layer.add_column("ww_softrank", value)

            value = np.max(evals)-np.max(rand_evals)
            ww_layer.add_column("ww_maxdist", value)

        if params[PLOT]:
            self.plot_random_esd(ww_layer, params)
            
        return ww_layer
    
    
    def apply_permute_W(self, ww_layer, params=None):
        """Randomize the layer weight matrices by using a deterministic permutation
        This will replace the Wmats ; they can be recovered by apply_unpermute_W()
         """
         
        if params is None: params = DEFAULT_PARAMS.copy()
        
        layer_id = ww_layer.layer_id
        plot_id =  ww_layer.plot_id
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
    
    
      
    def apply_unpermute_W(self, ww_layer, params=None):
        """Unpermute the layer weight matrices after the deterministic permutation
        This will replace the Wmats ; only works if applied after  apply_permute_W()
         """
        
        if params is None: params = DEFAULT_PARAMS.copy()
        
        layer_id = ww_layer.layer_id
        plot_id =  ww_layer.plot_id
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
    def apply_detX(self, ww_layer, params=None):
        """Compute the detX constraint, and optionally plot """
                    
        if params is None: params = DEFAULT_PARAMS.copy()
        
        plot = params[PLOT]           
        savefig = params[SAVEFIG]
        savedir = params[SAVEDIR]

        evals = ww_layer.evals        
        evals, Wscale = rescale_eigenvalues(evals)
        detX_num, detX_idx = detX_constraint(evals, rescale=False)
        detX_val = evals[detX_idx]


        if plot:
            name = ww_layer.name
            # fix rescaling to plot xmin

            layer_id = ww_layer.layer_id  # where is the layer_id
            plot_id =  ww_layer.plot_id
            plt.title(f"DetX constraint for {name}")
            plt.xlabel("log10 eigenvalues (norm scaled)")
            nz_evals = evals[evals > EVALS_THRESH]
            plt.hist(np.log10(nz_evals), bins=100)
            plt.axvline(np.log10(detX_val), color='purple', label=r"detX$=1$")
            
            if ww_layer.xmin:
                xmin = ww_layer.xmin *  np.max(evals)/ww_layer.xmax
                plt.axvline(np.log10(xmin), color='red', label=r"PL $\lambda_{min}$")
                
            plt.legend()
            if savefig:
                save_fig(plt, "detX", plot_id, savedir)
            plt.show(); plt.clf()

        ww_layer.add_column('detX_num', detX_num)

        detX_val_unrescaled = evals[detX_idx]

        evals = un_rescale_eigenvalues(evals, Wscale)
        detX_val = evals[detX_idx]

        ww_layer.add_column('detX_val', detX_val)
        ww_layer.add_column('detX_val_unrescaled', detX_val_unrescaled)

        return ww_layer
    
    
    # Not used yet
    def apply_plot_esd(self, ww_layer, params=None):
        """Plot the ESD on regular and log scale.  Only used when powerlaw fit not called"""
                    
        if params is None: params = DEFAULT_PARAMS.copy()
        
        evals = ww_layer.evals
        name = ww_layer.name
        
        plt.title(name)
        plt.hist(evals, bins=100)
        plt.show(); plt.clf()
        
        plt.title(name)
        plt.hist(np.log10(evals), bins=100)
        plt.show(); plt.clf()
            
        return ww_layer
    
     
    
 
    def apply_powerlaw(self, ww_layer, params=None):
        """Plot the ESD on regular and log scale.  Only used when powerlaw fit not called"""
                
        if params is None: params = DEFAULT_PARAMS.copy()
        
        evals = ww_layer.evals
        layer_id = ww_layer.layer_id
        plot_id =  ww_layer.plot_id
        name = ww_layer.name
        title = "{} {}".format(layer_id, name)

        xmin = None  # TODO: allow other xmin settings
        xmax = params[XMAX]#issue  199np.max(evals)
        plot = params[PLOT]
        sample = False  # TODO:  decide if we want sampling for large evals       
        sample_size = None

        savefig = params[SAVEFIG]
        savedir = params[SAVEDIR]

        fix_fingers =  params[FIX_FINGERS]
        xmin_max = params[XMIN_MAX]
        max_fingers =  params[MAX_FINGERS]
        finger_thresh = params[FINGER_THRESH]
        
        layer_name = "Layer {}".format(plot_id)
        
        fit_type =  params[FIT]
        pl_package = params[PL_PACKAGE]

        alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning = \
            self.fit_powerlaw(evals, xmin=xmin, xmax=xmax, plot=plot, layer_name=layer_name, layer_id=layer_id, \
                              plot_id=plot_id, sample=sample, sample_size=sample_size, savedir=savedir, savefig=savefig,  \
                              fix_fingers=fix_fingers, xmin_max=xmin_max, max_fingers=max_fingers, finger_thresh=finger_thresh, \
                              fit_type=fit_type, pl_package=pl_package)

  
        
        ww_layer.add_column('alpha', alpha)
        ww_layer.add_column('xmin', xmin)
        ww_layer.add_column('xmax', xmax)
        ww_layer.add_column('D', D)
        ww_layer.add_column('sigma', sigma)
        ww_layer.add_column('num_pl_spikes', num_pl_spikes)
        #ww_layer.add_column('best_fit', best_fit) 
        #ww_layer.add_column('fit_entropy', fit_entropy) #-1 for PL, 

        if fit_type==TPL or fit_type==TRUNCATED_POWER_LAW or fit_type==E_TPL:
            ww_layer.add_column('Lambda', Lambda)  

        if fix_fingers==CLIP_XMAX:
            ww_layer.add_column('num_fingers', num_fingers) 
            ww_layer.add_column('raw_alpha', raw_alpha) 

   
        ww_layer.add_column('status', status)
        ww_layer.add_column('warning', warning)

        return ww_layer


    def make_delta_layer_iterator(self, base_model, model=None, filters=[], params=None, Iterator_Class=WWLayerIterator):
        """make an iterator that takes returns the model weights (and biases) - base nmodel weights (and baises)"""
  
        framework = self.infer_framework(model)
        base_framework = self.infer_framework(base_model)
        
        delta_iter = WWDeltaLayerIterator(base_model=base_model, base_framework=base_framework, model=model, framework=framework, filters=filters, params=params, Iterator_Class=Iterator_Class)
        return delta_iter
    
    
    def make_peft_layer_iterator(self, base_model, model=None, filters=[], params=None, Iterator_Class=WWLayerIterator):
        """make an iterator that re-works the LORA A abd B layers to analyze with weightwatcher """
  
        framework = self.infer_framework(model)
        base_framework = self.infer_framework(base_model)
        
        peft_iter = WWDeltaLayerIterator(model=model, framework=framework, filters=filters, params=params, Iterator_Class=Iterator_Class)
        return peft_iter
    
    

    def make_layer_iterator(self, model=None, layers=[], params=None, base_model=None):
        """Constructor for the Layer Iterator; See analyze(...)
         """
         
        if params is None: params = DEFAULT_PARAMS.copy()
        self.set_model_(model, base_model=base_model)
            
        logger.info("params {}".format(params))
        if not WeightWatcher.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)

        #stacked = params['stacked']
        intra = params[INTRA]
        pool = params[POOL]
        stacked = params[STACKED]
        peft = params[PEFT]
        
        layer_iterator = None
        
        if stacked:
            logger.info("Using Stacked Iterator (experimental)")
            Iterator_Class = WWStackedLayerIterator
        elif intra:
            logger.info("using Intra layer Analysis (experimental)")
            Iterator_Class = WWIntraLayerIterator    
        elif peft:
            logger.info("using PEFT LORA layer Analysis (experimental)")
            Iterator_Class = WWPeftLayerIterator 
        elif not pool:
            logger.info("Pooling eigenvalues, Using weightwatcher 0.2x style layer and slice iterator")
            Iterator_Class = WW2xSliceIterator 
        else:
            Iterator_Class = WWLayerIterator 
        
   
        if base_model is None:
            logger.info(f"Using {Iterator_Class} ")
            layer_iterator = Iterator_Class(self.model, self.framework, filters=layers, params=params)    
        else:
            logger.info(f"Using Delta Layer Iterator with base_model = {base_model} and class = {Iterator_Class} ")
            layer_iterator = self.make_delta_layer_iterator(base_model, model=model, filters=layers, params=params, Iterator_Class=WWLayerIterator)
            
        return layer_iterator
    
    
    def vector_metrics(self, vectors=None, params=None): 
        """Analyze arbitrary vectors using random matrix theory RMT and report vector metrics
        
        Parameters
        ----------
        
        vectors:  a single numpy vector, list of vectors, or an (NxM)  numpy array
                  if a numpy array, expects N vectors of length M
        
        Returns
        ----------
        
        metrics dataframe with columns 
        (as described in the JMLR paper:  https://jmlr.org/papers/v22/20-410.html)
        
        """
        
        if params is None: params = DEFAULT_PARAMS.copy()

        df = pd.DataFrame(columns=["length", "entropy", "discrete_entropy", "localization_ratio", "participation_ratio"])

        if not WeightWatcher.valid_vectors(vectors):
            logger.warning("vectors not specified correctly, returning -1")
            return -1

        for iv, vec in enumerate(self.iterate_vectors(vectors)):
            df.loc[iv] = None
            df.at[iv, "length"] = len(vec)
            df.at[iv, "entropy"] = vector_entropy(vec)
            df.at[iv, "discete_entropy"] = discrete_entropy(vec)
            df.at[iv, "localization_ratio"] = localization_ratio(vec)
            df.at[iv, "participation_ratio"] = participation_ratio(vec)

        return df




    def iterate_vectors(self, vectors):
        """vectors may be a single vector, list of vectors, or a numoy array"""

        if isinstance(vectors, np.ndarray) and len(vectors.shape) > 1:
            # vectors is a numpy array, so iterate over rows
            # Get the number of rows and columns in the array
            num_rows, num_cols = vectors.shape

            # Iterate over the rows in the array
            for i in range(num_rows):
                # Get the current row
                row = vectors[i, :]
                yield row
            
        elif isinstance(vectors, list):
            # vectors is a list of numpy vectors
            for i, vec in enumerate(vectors):
                yield vec

        else:
            # vectors is a single numpy vector
            yield vectors

    @staticmethod
    def valid_vectors(vectors):
        """Returns true if the input is 1 or more numpy vectors, 
        either as  a single vector, a list of vectors, or a numpy array of vectors, 
        and returns false otherwise"""

        if isinstance(vectors, np.ndarray):
            # vectors is a numpy array, so we will flatten it to a list
            vectors = vectors.flatten()
            
        if isinstance(vectors, list):
            # vectors is a list of numpy vectors
            for vec in vectors:
                if not isinstance(vec, np.ndarray):
                    # at least one element of the list is not a numpy vector
                    return False
        elif not isinstance(vectors, np.ndarray):
            # vectors is not a single numpy vector or a list of numpy vectors
            return False
                    
        # vectors is a single numpy vector or a list of numpy vectors
        return True
        

    def analyze(self, model=None, layers=[], 
                min_evals=DEFAULT_MIN_EVALS, max_evals=DEFAULT_MAX_EVALS,
                min_size=None, max_size=None, max_N=DEFAULT_MAX_N,
                glorot_fix=False,
                plot=False, randomize=False,  
                savefig=DEF_SAVE_DIR,
                mp_fit=False, conv2d_norm=True,  
                ww2x=DEFAULT_WW2X, pool=DEFAULT_POOL,
                conv2d_fft=False, fft=False, 
                deltas=False, intra=False, vectors=False, channels=None, 
                stacked=False,
                fix_fingers=False, xmin_max = None,  max_fingers=DEFAULT_MAX_FINGERS, finger_thresh=DEFAULT_FINGER_THRESH,
                fit=PL, sparsify=True, 
                detX=False,
                svd_method=FAST_SVD,
                tolerance=WEAK_RANK_LOSS_TOLERANCE,
                start_ids=DEFAULT_START_ID,
                pl_package=WW_POWERLAW_PACKAGE,
                xmax=DEFAULT_XMAX,
                base_model=None,
                peft = DEFAULT_PEFT,
                inverse = False
                ):
        """
        Analyze the weight matrices of a model.

        Parameters
        ----------
        
        layers:
            List of layer ids. If empty, analyze all layers (default)
            If layer ids < 0, then skip the layers specified
            All layer ids must be > 0 or < 0
            
        min_evals:  int, default=10, NOT 0
            Minimum number of evals (M*rf) 
            
        max_evals:  int, default=15000
            Maximum number of evals (N*rf) (0 = no limit)
            
        max_Ns:  int, default=50000
            Maximum N, largest size of matrix
            
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
            
        
            
        ww2x:  bool, default: False (deprecated)
            Use weightwatcher version 0.2x style iterator, which slices up Conv2D layers in N=rf matrices
            This option is deprecated, please use pool = not ww2x
            
        conv2d_fft:  (deprecated)
            For Conv2D layers, apply FFT to the kernels.  O
            
            
        pool: bool, default: True
            For layers with multiple matrices (like COnv2D layers), pools the eigenvalues beforer running the  analysis
            
        fft:  (experimental)
            For Conv2D layers, apply the FFT method to the inpuyt/output maps (weight matrices) 
            Can be used with or without pooling
                
            
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
            
        fix_fingers:  False \'xmin_peak' | 'clip_xmax', default: False 
            Attempts to fix the fingers / finite-size-effects which sometimes occurs fpr layers with spuriously large alphas
            Can be very slow.
            
            Also, currently only 'xmin_peak' works, and you need to set the top of the xmin range specificalluy
            
        xmin_max = None | max eignalvaue in the xmin range
            Only makes sense to use with fix_fingers= 'xmin_fit'
            
        max_fingers: 10 by default, 
            Only makes sense to use with fix_fingers= 'clip_xmax'
            Max Number of eigenvalues to clip to find a much smaller alpha
            
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

        svd_method:  string, default: 'fast'
            Must be one of "fast" or "accurate". Determines the method by which eigenvalues are calcualted.
            
        tolerance: float, default 0.000001
            sets 'weak_rank_loss' = number of  eigenvalues <= tolerance
          
        params:  N/A yet
            a dictionary of default parameters, which can be set but will be over-written by 

        start_ids:  0 (default) | 1
           Start layer id counter at 0 or 1
           Only can be reset for PyTorch models, used when the layer_id=0 is undesirable
           
        powerlaw_package: 'ww' (default) | 'powerlaw'
           Lets users reset the powerlaw package to the older version on pypi
           
        xmax: None | -1 | -2 ... |  'force'
           If 'force', resets the powerlaw_package to 'powerlaw'
           if an int (-1, -2), ignores the top N eigenvalues
           
           Must be set to use fix_fingers, TPL, E_TPL
           Can not set 'force' and use the new default powerlaw_package: 'ww' 
           
           Setting to 'force' may cause the powerlw fits to find undesirable local minima, which can induce fingers
           
        base_model:  None | same model with different weights and biases
            if specified, the base model weights and biases are subtracted from the model
            used for analyzing fine-tuned LLMs, or just looking at the deltas (i.e from init) during training
            
        peft: set to True if using an PEFT / LORA fine-tuned model
        
        inverse:  compute the inverse ESD 
           
        """
        
        self.set_model_(model, base_model)    
        
        if min_size or max_size:
            logger.warning("min_size and max_size options changed to min_evals, max_evals, ignored for now")     
        
        
        if ww2x:
            logger.warning("WW2X option deprecated, reverting too POOL=False")
            ww2x=False
            pool=False
            
        params=DEFAULT_PARAMS.copy()          
        
        params[MIN_EVALS] = min_evals 
        params[MAX_EVALS] = max_evals
        params[MAX_N] = max_N
        
        params[PLOT] = plot
        params[RANDOMIZE] = randomize
        params[MP_FIT] = mp_fit
        #params[NORMALIZE] = normalize   #removed 0.6.5
        params[GLOROT_FIT] = glorot_fix
        params[CONV2D_NORM] = conv2d_norm
        
        params[POOL] = pool  
        #deprecated
        params[WW2X] = ww2x   
        params[CONV2D_FFT] = conv2d_fft

        #experimental
        params[FFT] = fft

        params[DELTA_ES] = deltas 
        params[INTRA] = intra 
        params[CHANNELS_STR] = channels
        params[LAYERS] = layers
        params[VECTORS] = vectors
        params[STACKED] = stacked
        params[FIX_FINGERS] = fix_fingers
        params[XMIN_MAX] = xmin_max
        params[MAX_FINGERS] = max_fingers
        params[FINGER_THRESH] = finger_thresh

        params[FIT] = fit
        params[SPARSIFY] = sparsify
        params[DETX] = detX
        params[SVD_METHOD] = svd_method
        params[TOLERANCE] = tolerance
        params[START_IDS] = start_ids


        params[SAVEFIG] = savefig
        #params[SAVEDIR] = savedir
        
        params[PL_PACKAGE] = pl_package
        params[XMAX] = xmax
        
        params[PEFT] = peft
        params[INVERSE] = inverse


            
        logger.debug("params {}".format(params))
        if not WeightWatcher.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)
        
        
        #TODO: ? 
        #  if we want deltas, we can just add the delta="model" in 
        # 
        layer_iterator = self.make_layer_iterator(model=self.model, layers=layers, params=params, base_model=self.base_model)     
        
        details = pd.DataFrame(columns=[])

        for ww_layer in layer_iterator:
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.debug("LAYER: {} {}  : {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.framework_layer)))
                
                # maybe not necessary
                self.apply_normalize_Wmats(ww_layer, params)
                
                # TODO: dd apply_fft
               
                if params[FFT]:
                     self.apply_FFT(ww_layer, params)
                    
                self.apply_esd(ww_layer, params)
                
                
                if ww_layer.evals is not None:
                    self.apply_powerlaw(ww_layer, params)
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
                            logger.debug("Computing and Plotting Deltas: {} {} ".format(ww_layer.layer_id, ww_layer.name))
                            self.apply_plot_deltaEs(ww_layer, random=True, params=params)
                        
                    if params[DETX]:
                        logger.debug("Finding detX constaint: {} {} ".format(ww_layer.layer_id, ww_layer.name)) 
                        self.apply_detX(ww_layer, params=params)
                    
                    self.apply_norm_metrics(ww_layer, params)
                    #all_evals.extend(ww_layer.evals)
                    
                # TODO: add find correlation traps here
                                
                # issue 137 (moved code here)
                # details = details.append(ww_layer.get_row(), ignore_index=True)
                data = pd.DataFrame.from_records(ww_layer.get_row() , index=[0])
                details = pd.concat([details,data], ignore_index=True)

        # Reorder the columns so that layer_id and name come first.
        if len(details) > 0:
            lead_cols = ["layer_id", "name"]
            details = details[lead_cols + [c for c in details.columns if not c in lead_cols]]

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

    def set_model_(self, model, base_model=None):
        """Set the model if it has not been set for this object
        
        maybe we should read the config file here ?  maybe user should specify the config file ?"""
        
        logger.info(f"SET MODEL: {model}, {base_model}")
        self.model = model or self.model
        if self.model is None:
            logger.fatal("unknown model, stopping")
                        
        if model is not None and self.base_model is None:
            self.base_model = base_model
            
        if self.framework is None:
            self.framework = self.infer_framework(self.model) 
            if not WeightWatcher.valid_framework(self.framework):
                logger.fatal(f"{self.framework} is not a valid framework, stopping")
      
        
        if self.base_model is not None:
            base_framework = self.infer_framework(self.base_model) 
            if not WeightWatcher.valid_framework(self.framework):
                logger.fatal(f"{base_framework} is not a valid base_framework, stopping")
                
            if base_framework!=self.framework:
                 logger.fatal(f"base_framework {base_framework} is different from model framework {self.framework}, stopping")
                 
        #issue #243
        if self.framework==FRAMEWORK.PYSTATEDICT:
            if 'state_dict' in self.model:
                self.model = self.model['state_dict']
                
            if self.base_model is not None:
                if 'state_dict' in self.base_model:
                    self.model = self.base_model['state_dict']
                 
        return 
    
    
                
    # test with https://github.com/osmr/imgclsmob/blob/master/README.md
    def describe(self, model=None, layers=[], min_evals=DEFAULT_MIN_EVALS, max_evals=DEFAULT_MAX_EVALS,
                min_size=None, max_size=None, max_N=DEFAULT_MAX_N,
                glorot_fix=False, 
                savefig=DEF_SAVE_DIR, ww2x=False, pool=True,
                conv2d_fft=False,  fft=False, conv2d_norm=True, 
                intra=False, channels=None, stacked=False,  start_ids=0,                 
                base_model=None, peft=DEFAULT_PEFT):
        """
        Same as analyze() , but does not run the ESD or Power law fits
        
        BUT min_evals default here is 0, not DEFAULT_MIN_EVALS = 50
        Not great...we need to fix
        
        
        """

        self.set_model_(model, base_model)    
        
        if min_size or max_size:
            logger.warning("min_size and max_size options changed to min_evals, max_evals, ignored for now")     

        if ww2x:
            logger.warning("WW2X option deprecated, reverting too POOL=False")
            ww2x=False
            pool=False
            
        params = DEFAULT_PARAMS.copy()

        params[MIN_EVALS] = min_evals 
        params[MAX_EVALS] = max_evals
        params[MAX_N] = max_N
      
        # params[NORMALIZE] = normalize  #removed 0.6.5 
        params[GLOROT_FIT] = glorot_fix
        params[CONV2D_NORM] = conv2d_norm
        
        params[CONV2D_FFT] = conv2d_fft
        params[WW2X] = ww2x   
        params[POOL] = pool   

        params[FFT] = fft
        
        params[INTRA] = intra 
        params[CHANNELS_STR] = channels
        params[LAYERS] = layers
        params[STACKED] = stacked
        
        params[SAVEFIG] = savefig
        #params[SAVEDIR] = savedir
        params[START_IDS] = start_ids
        
        params[PEFT] = peft

        
        logger.info("params {}".format(params))
        if not WeightWatcher.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)

        layer_iterator = self.make_layer_iterator(model=self.model, layers=layers, params=params, base_model=self.base_model)     
        details = pd.DataFrame(columns=[])
           
        num_all_evals = 0
        for ww_layer in layer_iterator:
            if not ww_layer.skipped and ww_layer.has_weights:
                #logger.debug("LAYER TYPE: {} {}  layer type {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.framework_layer)))
                #logger.debug("layer_id {} weights shape : {}  max size {}".format(ww_layer.layer_id,ww_layer.weights.shape, params['max_evals']))
                if not pool:
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
                details = pd.concat([details,data], ignore_index=True)

        if len(details)>0:
        # Reorder the columns so that layer_id and name come first.
            lead_cols = ["layer_id", "name"]
            details = details[lead_cols + [c for c in details.columns if not c in lead_cols]]
        else:
            logger.warning("No layers found")
        return details

    @staticmethod
    def valid_params(params):
        """Validate the input parameters, return True if valid, False otherwise"""
        
        valid = True        
        xmin = params.get('xmin')

        if xmin and xmin not in [XMIN.UNKNOWN, XMIN.AUTO, XMIN.PEAK]:
            logger.warning("param xmin unknown, ignoring {}".format(xmin))
            valid = False
            
        min_evals = params.get(MIN_EVALS) 
        max_evals = params.get(MAX_EVALS)
        max_N = params.get(MAX_N)
        
        if min_evals and max_evals and min_evals >= max_evals:
            logger.warning("min_evals {} > max_evals {}".format(min_evals, max_evals))
            valid = False
        elif max_evals and max_evals < -1:
            logger.warning(" max_evals {} < -1 ".format(max_evals))
            valid = False
        
        if  max_N < max_evals:
            logger.warning(f" max_N {max_N} < max_evals {max_evals}")
            valid = False
            
        if max_N < min_evals:
            logger.warning(f" max_N {max_N} < min_evals {max_evals}")
            valid = False

        svd_method = params.get(SVD_METHOD)
        if svd_method not in VALID_SVD_METHODS:
            logger.warning("unrecognized svd_method {}. Must be one of {}".format(svd_method, VALID_SVD_METHODS))
            valid = False
            
        
        if params.get(WW2X):
            logger.warning("ww2x option deprecated, please use pool=false")
            valid = False

            
        # can not specify ww2x and conv2d_fft at same time
        # NOT SURE ABOUT THIS
        if not params.get(POOL) and params.get('conv2d_fft'):
            logger.warning("can not specify conv2d_fft without pool=True")
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

        # layer can be an  list of all + or all -  integers
        # eventually this to be exteneisvly united tested
        filters = params.get(LAYERS) 
        if filters is not None:
            if isinstance(filters, numbers.Integral):
                filters = [filters]
            elif isinstance(filters, np.ndarray):
                filters = filters.tolist()
          
            filter_ids = [int(f) for f in filters if isinstance(f, numbers.Integral)]

            if len(filter_ids) > 0:
                if np.max(filter_ids) > 0 and np.min(filter_ids) < 0:
                    logger.warning("layer filter ids must be all > 0 or < 0: {}".format(filter_ids))
                    valid = False
         
        savefig = params.get(SAVEFIG)
        savedir = params.get(SAVEDIR)
        if savefig and isinstance(savefig,bool):
            logger.info("Saving all images to {}".format(savedir))
        elif savefig and isinstance(savefig,str):
            # noteL this is a global change, not ideal but I think its ok
            params[SAVEDIR] = savefig
            logger.info("Saving all images to {}".format(savedir))
        elif not isinstance(savefig,str) and not isinstance(savefig,bool):
            valid = False      
            
            
        fix_fingers =  params[FIX_FINGERS]
        max_fingers =  params[MAX_FINGERS]
        xmax = params[XMAX]
        if fix_fingers:
            if fix_fingers not in [XMIN_PEAK, CLIP_XMAX]:
                logger.warning(f"Unknown how to fix fingers {fix_fingers}, deactivating")
                valid=False
            elif xmax is not None and xmax is not False and isinstance(xmax,int):
                logger.warning(f"Can not set fix fingers, with xmax ={xmax}")
                valid=False
            else:
                logger.info("Fixing fingers using  {}".format(fix_fingers))
                
            if max_fingers is not None and max_fingers < 1:
                logger.warning(f"Can not set max fingers < 1, max_fingers={max_fingers}")
                valid=False
                
            
        fit_type = params[FIT]
        if fit_type not in [PL, TPL, E_TPL, POWER_LAW, TRUNCATED_POWER_LAW]:
            logger.warning("Unknown fit type {}".format(fit_type))
            valid = False
            
        if fit_type==E_TPL and fix_fingers==CLIP_XMAX:
            logger.warning(f"E-TPL and fix_fingers ={CLIP_XMAX} can not both be explicitly set")
            valid = False


        intra = params[INTRA]
        if intra:
            if params[RANDOMIZE] or params[VECTORS]:
                logger.fatal("Can not set intra=True with randomize=True or vectors=True at this time")
                valid = False

        start_ids = params[START_IDS]
        if start_ids < 0: #not in [0,1]:
            logger.fatal(f"Layer Ids must start at 0 or 1, start_ids={start_ids}")
            valid = False
            
        pl_package = params[PL_PACKAGE]
        if pl_package not in [POWERLAW_PACKAGE, WW_POWERLAW_PACKAGE]:
            logger.fatal(f"Powerlaw Package not valid={pl_package}, default is {DEFAULT_POWERLAW_PACKAGE}")
            valid = False
            
        xmax = params[XMAX]
        if not isinstance(xmax, int) and xmax not in [False, None, 'force']:
            logger.fatal(f"xmax must be None or 'force', xmax={xmax}")
            valid = False
        elif isinstance(xmax, int) and xmax==0:
            logger.fatal(f"xmax can not be 0")
            valid = False
                            
        if pl_package==WW_POWERLAW_PACKAGE:
            if xmax=='force':
                logger.fatal(f"xmax=force not available if using {PL_PACKAGE}={WW_POWERLAW_PACKAGE}")
                valid = False
            if fit_type not in [PL, POWER_LAW]:
                logger.fatal(f"{PL_PACKAGE} only supports PowerLaw fits, but {FIT}={fit_type}")
                valid = False
                
        #if ((xmax is None) or (xmax is False) or (xmax!=FORCE) or (isinstance(xmax, int))) and fix_fingers in [XMIN_PEAK, CLIP_XMAX]:
        #    logger.warning(f"{FIX_FINGERS} ignores xmax = {xmax}" )
        #    valid = True
            
            
        inverse = params[INVERSE]
        if inverse:
            for theParam in [VECTORS]:
                if params[theParam]:
                    logger.fatal(f"Sorry, the {theParam} option is not yet available when using inverse=True")
                    valid = False
                
        return valid
    
    
    def normalize_params(self, params):
        """Reset params to allow some syntactic sugar in the inputs"""
        
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
    
    def random_eigenvalues(self, Wmats, n_comp, num_replicas=1, params=None):
        """Compute the eigenvalues for all weights of the NxM skipping layer, num evals sized weight matrices (N >= M), 
            combined into a single, sorted, numpy array.  
    
        see: combined_eigenvalues()
        
         """
        
        if params is None: params = DEFAULT_PARAMS.copy()
         
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
                sv = svd_vals(W, method=params[SVD_METHOD])
                sv = sv.flatten()
                sv = np.sort(sv)[-n_comp:]    
                
                if params[INTRA]:
                    evals = sv
                    sv = np.sqrt(sv)
                else:
                    evals = sv * sv 
                all_evals.extend(evals)
                
        if params[INVERSE]:
            all_evals = 1.0/np.array(all_evals)
        else:
            all_evals = np.array(all_evals)
                                       
        return np.sort(all_evals)
   
    def plot_random_esd(self, ww_layer, params=None):
        """Plot histogram and log histogram of ESD and randomized ESD"""
          
        if params is None: params = DEFAULT_PARAMS.copy()
        
        savefig = params[SAVEFIG]
        savedir = params[SAVEDIR]

        layer_id = ww_layer.layer_id
        plot_id = ww_layer.plot_id
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
            save_fig(plt, "randesd1", plot_id, savedir)
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
            save_fig(plt, "randesd2", plot_id, savedir)
        plt.show(); plt.clf()
        

    def fit_powerlaw(self, evals, xmin=None, xmax=None, plot=True, layer_name="", layer_id=0, plot_id=0, \
                     sample=False, sample_size=None,  savedir=DEF_SAVE_DIR, savefig=True, \
                     thresh=EVALS_THRESH,\
                     fix_fingers=False, finger_thresh=DEFAULT_FINGER_THRESH, xmin_max=None, max_fingers=DEFAULT_MAX_FINGERS, \
                     fit_type=POWER_LAW, pl_package=WW_POWERLAW_PACKAGE):
        """Fit eigenvalues to powerlaw or truncated_power_law
        
            if xmin is 
                'auto' or None, , automatically set this with powerlaw method
                'peak' , try to set by finding the peak of the ESD on a log scale
            
            if xmax is 'auto' or None, xmax = np.max(evals)
            
            svd_method = ACCURATE_SVD (to add TRUNCATED_SVD with some cutoff)
            thresh is a threshold on the evals, to be used for very large matrices with lots of zeros
            
                     
         """
         
        status = None
        
        # defaults for failed status
        alpha = -1
        Lambda = -1
        D = -1
        sigma = -1
        xmin = -1  # not set / error
        num_pl_spikes = -1
        best_fit = UNKNOWN
        fit = None
        num_fingers = 0
        
        raw_fit = None # only for fix fingers
        
        fit_entropy = -1
        
        # check    
        num_evals = len(evals)
        logger.debug("fitting {} on {} eigenvalues".format(fit_type, num_evals))

        # if num_evals < MIN_NUM_EVALS:  # 10 , really 50
        #     logger.warning("not enough eigenvalues, stopping")
        #     status = FAILED
        #     return alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, best_fit, status
                          
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
                    
        if xmax == XMAX_FORCE:
            logger.info("forcing xmax, alpha may be over-estimated")
            xmax = np.max(evals)
        elif isinstance(xmax, int):
            xmax = np.abs(xmax)
            if xmax > len(evals)/2:
                logger.warning("xmax is too large, stopping")
                status = FAILED
            else:
                logger.info(f"clipping off {xmax} top eigenvalues")
                evals = evals[:-xmax] 
                xmax = None
        else:
            logger.debug("xmax not set, fast PL method in place")
            xmax = None
                        
        if fix_fingers==XMIN_PEAK:
            logger.info("fix the fingers by setting xmin to the peak of the ESD")
            try:
                nz_evals = evals[evals > thresh]
                num_bins = 100  # np.min([100, len(nz_evals)])
                h = np.histogram(np.log10(nz_evals), bins=num_bins)
                ih = np.argmax(h[0])
                xmin2 = 10 ** h[1][ih]
                if xmin_max is None:
                    xmin_max = 1.5 * xmin2 
                elif xmin_max <  0.95 * xmin2:
                    logger.fatal("XMIN max is too small, stopping  ")  
                    
                if pl_package!=POWERLAW_PACKAGE:
                    logger.fatal(f"Only  {POWERLAW_PACKAGE} supports fix_fingers, pl_package mis-specified, {pl_package}")
                    
                xmin_range = (np.log10(0.95 * xmin2), xmin_max)
                logger.info(f"using new XMIN RANGE {xmin_range}")
                fit = pl_fit(data=nz_evals, xmin=xmin_range, xmax=xmax, verbose=False, distribution=distribution, pl_package=pl_package)  
                status = SUCCESS 
            except ValueError as err:
                logger.warning(str(err))
                status = FAILED
            except Exception as err:
                logger.warning(str(err))
                status = FAILED
                
        elif fix_fingers==CLIP_XMAX:
            logger.info(f"fix the fingers by fitting a clipped power law using pl_package = {pl_package}, xmax={xmax}")
            try:
                nz_evals = evals[evals > thresh]
                if max_fingers is None or max_fingers < 0 or max_fingers < (1/2)*len(evals):
                    max_fingers = DEFAULT_MAX_FINGERS
                logger.debug(f"max_fingers = {MAX_FINGERS}")
                    
                fit, num_fingers, raw_fit = fit_clipped_powerlaw(nz_evals, xmax=xmax, max_fingers=max_fingers, finger_thresh=finger_thresh, \
                                                        logger=logger, plot=plot,  pl_package=pl_package)  
                status = SUCCESS 
            except ValueError as err:
                logger.warning(str(err))
                status = FAILED
            except Exception as err:
                logger.warning(str(err))
                status = FAILED
             
        elif xmin is None or xmin == -1: 
            logger.info(f"Running powerlaw.Fit no xmin, xmax={xmax}. distribution={distribution} pl_package={pl_package}")
            try:
                nz_evals = evals[evals > thresh]
                fit = pl_fit(data=nz_evals, xmax=xmax, verbose=False, distribution=distribution, pl_package=pl_package) 
                status = SUCCESS 
            except ValueError as err:
                logger.warning(str(err))
                status = FAILED
            except Exception as err:
                logger.warning(str(err))
                status = FAILED

        else: 
            #logger.debug("POWERLAW DEFAULT XMIN SET ")
            try:
                fit = pl_fit(data=evals, xmin=xmin,  verbose=False, distribution=distribution, pl_package=pl_package)  
                status = SUCCESS 
            except ValueError as err:
                logger.warning(str(err))
                status = FAILED
            except Exception as  err:
                logger.warning(str(err))
                status = FAILED
                    
        if fit is None or fit.alpha is None or np.isnan(fit.alpha):
            logger.warning("Power law fit failed")

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
                
            #logger.debug("finding best distribution for fit, TPL or other ?")
            # we stil check againsgt TPL, even if using PL fit
            # all_dists = [TRUNCATED_POWER_LAW, POWER_LAW, LOG_NORMAL]#, EXPONENTIAL]
            # Rs = [0.0]
            # dists = [POWER_LAW]
            # for dist in all_dists[1:]:
            #     R, p = pl_compare(fit, dist)
            #     if R > 0.1 and p > 0.05:
            #         dists.append(dist)
            #         Rs.append(R)
            #         logger.debug("compare dist={} R={:0.3f} p={:0.3f}".format(dist, R, p))
            # best_fit = dists[np.argmax(Rs)]
            
            #fit_entropy = line_entropy(fit.Ds)

        if plot:
            
            if status==SUCCESS:
                min_evals_to_plot = (xmin/100)
                
                fig2 = fit.plot_pdf(color='b', linewidth=0) # invisbile
                fig2 = fit.plot_pdf(color='r', linewidth=2)
                if fit_type==POWER_LAW:
                    if pl_package == WW_POWERLAW_PACKAGE:
                        fit.plot_power_law_pdf(color='r', linestyle='--', ax=fig2)
                    else:
                        fit.power_law.plot_pdf(color='r', linestyle='--', ax=fig2)
                
                else:
                    fit.truncated_power_law.plot_pdf(color='r', linestyle='--', ax=fig2)
            else:
                xmin = -1
                min_evals_to_plot = (0.4*np.max(evals)/100)

            evals_to_plot = evals[evals>min_evals_to_plot]
            plot_loghist(evals_to_plot, bins=100, xmin=xmin)
            title = "Log-Log ESD for {}\n".format(layer_name) 
            
            if status==SUCCESS:
                title = title + r"$\alpha=${0:.3f}; ".format(alpha) + \
                    r'$D_{KS}=$'+"{0:.3f}; ".format(D) + \
                    r"$\lambda_{min}=$"+"{0:.3f} ".format(xmin) + \
                    r"$\sigma=$"+"{0:.3f}".format(sigma) + "\n"
            else:
                title = title + " PL FIT FAILED"

            plt.title(title)
            plt.legend()
            if savefig:
                #plt.savefig("ww.layer{}.esd.png".format(layer_id))
                save_fig(plt, "esd", plot_id, savedir)
            plt.show(); plt.clf()
                
    
            # plot eigenvalue histogram
            num_bins = 100  # np.min([100,len(evals)])
            plt.hist(evals_to_plot, bins=num_bins, density=True)
            title = "Lin-Lin ESD for {}".format(layer_name) 
            plt.title(title)
            plt.axvline(x=fit.xmin, color='red', label=r'$\lambda_{xmin}$')
            plt.legend()
            if savefig:
                #plt.savefig("ww.layer{}.esd2.png".format(layer_id))
                save_fig(plt, "esd2", plot_id, savedir)
            plt.show(); plt.clf()

            # plot log eigenvalue histogram
            nonzero_evals = evals_to_plot[evals_to_plot > 0.0]
            plt.hist(np.log10(nonzero_evals), bins=100, density=True)
            title = "Log-Lin ESD for {}".format(layer_name) 
            plt.title(title)
            plt.axvline(x=np.log10(fit.xmin), color='red', label=r'$\lambda_{xmin}$')
            plt.axvline(x=np.log10(fit.xmax), color='orange',  label=r'$\lambda_{xmax}$')
            plt.legend()
            if savefig:
                #plt.savefig("ww.layer{}.esd3.png".format(layer_id))
                save_fig(plt, "esd3", plot_id, savedir)
            plt.show(); plt.clf()
    
            # plot xmins vs D
            
            plt.plot(fit.xmins, fit.Ds, label=r'$D_{KS}$')
            plt.axvline(x=fit.xmin, color='red', label=r'$\lambda_{xmin}$')
            #plt.plot(fit.xmins, fit.sigmas / fit.alphas, label=r'$\sigma /\alpha$', linestyle='--')
            plt.xlabel(r'$x_{min}$')
            plt.ylabel(r'$D_{KS}$')
            title = r'$D_{KS}$'+ ' vs.' + r'$x_{min},\;\lambda_{xmin}=$'
            plt.title(title+"{:0.3}".format(fit.xmin))
            plt.legend()

            #ax = plt.gca().twinx()
            #ax.plot(fit.xmins, fit.alphas, label=r'$\alpha(xmin)$', color='g')
            #ax.set_ylabel(r'$\alpha$')
            #ax.legend()
            
            if savefig:
                save_fig(plt, "esd4", plot_id, savedir)
                #plt.savefig("ww.layer{}.esd4.png".format(layer_id))
            plt.show(); plt.clf() 
            
            
            plt.plot(fit.xmins, fit.alphas, label=r'$\alpha(xmin)$')
            plt.axvline(x=fit.xmin, color='red', label=r'$\lambda_{xmin}$')
            plt.xlabel(r'$x_{min}$')
            plt.ylabel(r'$\alpha$')
            title = r'$\alpha$' + ' vs.' + r'$x_{min},\;\lambda_{xmin}=$'
            plt.title(title+"{:0.3}".format(fit.xmin))
            plt.legend()
            if savefig:
                save_fig(plt, "esd5", plot_id, savedir)
                #plt.savefig("ww.layer{}.esd5.png".format(layer_id))
                
                                
            plt.show(); plt.clf() 

        raw_alpha = -1
        if raw_fit is not None:
            raw_alpha = raw_fit.alpha
            
        warning = ""   
        if alpha < OVER_TRAINED_THRESH:
            warning = OVER_TRAINED
        elif alpha > UNDER_TRAINED_THRESH:
            warning = UNDER_TRAINED
            
        return alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning
    
    
    def get_ESD(self, model=None, layer=None, random=False, params=None):
        """Get the ESD (empirical spectral density) for the layer, specified by id or name)"""
        
        if params is None: params = DEFAULT_PARAMS.copy()
        
        self.set_model_(model)          
        
        details = self.describe(model=self.model)
        layer_ids = details['layer_id'].to_numpy()
        layer_names = details['name'].to_numpy()
        
        
        if isinstance(layer, numbers.Integral) and layer not in layer_ids:
            logger.error("Can not find layer id {} in valid layer_ids {}".format(layer, layer_ids))
            return []
        
        elif type(layer) is str and layer not in layer_names:
            logger.error("Can not find layer name {} in valid layer_names {}".format(layer, layer_names))
            return []
    
        layer_iter = WWLayerIterator(model=self.model, framework=self.framework, filters=[layer], params=params)     
        details = pd.DataFrame(columns=['layer_id', 'name'])
           
        ww_layer = next(layer_iter)
        assert(not ww_layer.skipped) 
        assert(ww_layer.has_weights)
        
        if not random:
            logger.info(f"Getting ESD for layer {layer} ; ww_layer id = {ww_layer.layer_id}")
            self.apply_esd(ww_layer, params)
            esd = ww_layer.evals
        else:
            logger.info(f"Getting Randomized ESD for layer {layer} ")
            self.apply_random_esd(ww_layer, params)
            esd = ww_layer.rand_evals

        if esd is None or len(esd)==0:
            logger.warning("No eigenvalues found for {} {}".format(ww_layer.layer_id, ww_layer.name))
                
        else:
            logger.debug("Found {} eiganvalues for {} {}".format(len(esd), ww_layer.layer_id, ww_layer.name))     
            
        return esd

    def get_Weights(self, model=None, layer=None, params=None):
        """Get the Weights for the layer, specified by id or name)"""
        
        if params is None: params = DEFAULT_PARAMS.copy()
        
        self.set_model_(model) 
        
        details = self.describe(model=self.model)
        layer_ids = details['layer_id'].to_numpy()
        layer_names = details['name'].to_numpy()
             
        if isinstance(layer, numbers.Integral) and layer not in layer_ids:
            logger.error("Can not find layer id {} in valid layer_ids {}".format(layer, layer_ids))
            return []
        
        elif type(layer) is str and layer not in layer_names:
            logger.error("Can not find layer name {} in valid layer_names {}".format(layer, layer_names))
            return []
    
        logger.info("Getting Weights for layer {} ".format(layer))

        layer_iter = WWLayerIterator(model=self.model, framework=self.framework, filters=[layer], params=params)     
        details = pd.DataFrame(columns=['layer_id', 'name'])
           
        ww_layer = next(layer_iter)
        assert(not ww_layer.skipped) 
        assert(ww_layer.has_weights)
        
        return ww_layer.Wmats
    
    
    def get_framework_layer(self, model=None, layer=None, params=None):
        """Get the underlying framework layer, specified by id or name)"""
        
        if params is None: params = DEFAULT_PARAMS.copy()
        
        self.set_model_(model) 
        
        details = self.describe(model=self.model)
        layer_ids = details['layer_id'].to_numpy()
        layer_names = details['name'].to_numpy()
             
        if isinstance(layer, numbers.Integral) and layer not in layer_ids:
            logger.error("Can not find layer id {} in valid layer_ids {}".format(layer, layer_ids))
            return []
        
        elif type(layer) is str and layer not in layer_names:
            logger.error("Can not find layer name {} in valid layer_names {}".format(layer, layer_names))
            return []
    
        logger.info("Getting Weights for layer {} ".format(layer))

        layer_iter = WWLayerIterator(model=self.model, framework=self.framework, filters=[layer], params=params)     
        details = pd.DataFrame(columns=['layer_id', 'name'])
           
        ww_layer = next(layer_iter)
        
        return ww_layer.framework_layer.layer
    
    
    def apply_norm_metrics(self, ww_layer, params=None):
        """Compute the norm metrics, as they depend on the eigenvalues"""

        if params is None: params = DEFAULT_PARAMS.copy()

        layer_id = ww_layer.layer_id
        plot_id = ww_layer.plot_id
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
    def apply_plot_deltaEs(self, ww_layer, random=False, params=None):
        """Plot the deltas of the layer ESD, both in a sequence as a histogram (level statisitcs)"""
        
        if params is None: params = DEFAULT_PARAMS.copy()
        
        layer_id = ww_layer.layer_i
        plot_id = ww_layer.plot_id
        name = ww_layer.name or ""
        layer_name = "{} {}".format(plot_id, name)
        
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
            save_fig(plt, "deltaEs", plot_id, savedir)
        plt.show(); plt.clf()

        
        # level statistics (not mean adjusted because plotting log)
        plt.hist(logDeltaEs, bins=100, color=color, density=True)
        plt.title("Log Level Statisitcs for Layer {}".format(layer_name))
        plt.ylabel("density")
        plt.xlabel(eqn)
        plt.legend()
        if savefig:  
            #plt.savefig("ww.layer{}.level-stats.png".format(layer_id))         
            save_fig(plt, "level-stats", plot_id, savedir)
        plt.show(); plt.clf()

    def apply_mp_fit(self, ww_layer, random=True, params=None):
        """Perform MP fit on random or actual random eigenvalues
        N/A yet"""

        if params is None: params = DEFAULT_PARAMS.copy()

        layer_id = ww_layer.layer_id
        plot_id = ww_layer.plot_id
        name = ww_layer.name or ""
        layer_name = "{} {}".format(plot_id, name)
        
        savefig = params[SAVEFIG]
        savedir = params[SAVEDIR]
        plot = params[PLOT]
        
        rescale = params[RESCALE] #should be True always
        
        if random:
            layer_name = "{} Randomized".format(layer_name)
            title = "Layer {} W".format(layer_name)
            evals = ww_layer.rand_evals
            orig_evals = ww_layer.evals

            color='mediumorchid'
        else:
            title = "Layer {} W".format(layer_name)
            evals = ww_layer.evals
            orig_evals = ww_layer.evals

            color='blue'

        N, M = ww_layer.N, ww_layer.M
        rf = ww_layer.rf

       # Initialize variables
        num_spikes, sigma_mp, mp_softrank, bulk_min, bulk_max, Wscale = -1, -1, -1, -1, -1, -1
        
        try:
            # Attempt to fit using mp_fit method
            num_spikes, sigma_mp, mp_softrank, bulk_min, bulk_max, Wscale = self.mp_fit(
                evals, N, M, rf, layer_name, layer_id, plot_id, plot, savefig, savedir, color, rescale, orig_evals
            )
        except Exception as e:
            # Handle any exception and check for specific types
            if isinstance(e, ValueError):
                logger.warning("A ValueError occurred during mp_fit.")
            elif isinstance(e, RuntimeError):
                logger.warning("A RuntimeError occurred during mp_fit.")
            else:
                logger.warning(f"An error occurred during mp_fit: {e}")
        
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
            ww_layer.add_column('mp_softrank', mp_softrank)
            ww_layer.add_column('W_scale', Wscale)
            ww_layer.add_column('bulk_max', bulk_max)
            ww_layer.add_column('bulk_min', bulk_min)
        return 

    def mp_fit(self, evals, N, M, rf, layer_name, layer_id, plot_id, plot, savefig, savedir, color, rescale, orig_evals):
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
                    save_fig(plt, "mpfit1", plot_id, savedir)
                plt.show(); plt.clf()
            
        else:
            fit_law = 'MP ESD'
#        
        #logger.info("MP fit min_esd={:0.2f}, max_esd={:0.2f}, Q={}, s1={:0.2f} Wsc ale={:0.2f}".format(np.min(to_plot), np.max(to_plot), Q, s1, Wscale))
        sigma_mp, x, mp = plot_density_and_fit(model=None, eigenvalues=to_plot, layer_name=layer_name, layer_id=0,
                              Q=Q, num_spikes=0, sigma=s1, verbose = False, plot=plot, color=color, cutoff=bulk_max_TW)#, scale=Wscale)
        
        if plot:
            title = fit_law +" for layer "+layer_name+"\n Q={:0.3} ".format(Q)
            title = title + r"$\sigma_{mp}=$"+"{:0.3} ".format(sigma_mp)
            title = title + r"$\mathcal{R}_{mp}=$"+"{:0.3} ".format(mp_softrank)
            title = title + r"$\#$ spikes={}".format(num_spikes)
            plt.title(title)
            
            if savefig:
                #plt.savefig("ww.layer{}.mpfit2.png".format(layer_id))
                save_fig(plt, "mpfit2", plot_id, savedir)
            plt.show(); plt.clf()
        
            # TODO: replot on log scale, along with randomized evals
            # we might add this back in later
            
            # plt.hist(to_plot, bins=100, density=True)
            # plt.hist(to_plot, bins=100, density=True, color='red')
            #
            # orig_plot = (Wscale*Wscale)*orig_evals.copy()
            # plt.hist(orig_plot[orig_plot<5], bins=100, density=True, color='green')
            #
            # plt.plot(x, mp, linewidth=1, color='r', label="MP fit")
            # plt.title("MP fit LOG PLOT  DEBUG")
            # plt.show()

        bulk_max = bulk_max/(Wscale*Wscale)
        bulk_min = bulk_min/(Wscale*Wscale)
        return num_spikes, sigma_mp, mp_softrank, bulk_min, bulk_max, Wscale
    
    
    def smooth_W(self, W, n_comp, svd_method):
        smoothed_W = smooth_W_switch(W, n_comp, svd_method)
        logger.info("smoothed W {} -> {} n_comp={} with {}".format(W.shape, smoothed_W.shape, n_comp, svd_method))
        return smoothed_W

        
    def smooth_W_alt(self, W, n_comp, svd_method):
        """Apply the SVD Smoothing Transform to W
        if n_comp < 0, then chomp off the top n_comp eiganvalues
        """       
        
        N, M = np.max(W.shape), np.min(W.shape)

        # TODO: replace this with truncated SVD
        # can't we just apply the svd transform...test
        # keep this old method for historical comparison
        u, s, vh = svd_full(W, method=svd_method)

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
    
    
        
                

    # AUTO:  should probably use fix fingers

    def SVDSmoothing(self, model=None, percent=0.8, pool=True, layers=[], method=DETX, fit=PL, plot=False, start_ids=0, svd_method=FAST_SVD):
        """Apply the SVD Smoothing Transform to model, keeping (percent)% of the eigenvalues
        
        layers:
            List of layer ids. If empty, analyze all layers (default)
            If layer ids < 0, then skip the layers specified
            All layer ids must be > 0 or < 0
        
        pool:
            pool the eigenvalues before applying powerlaw analysis
            if pool=False, Use weightwatcher version 0.2x style iterator, which slices up Conv2D layers in N=rf matrices
        
        """
        
        self.set_model_(model)          
         
        params = DEFAULT_PARAMS.copy()
        
        params[POOL] = pool
        params[LAYERS] = layers
        params[FIT] = fit # only useful for method=LAMBDA_MINa
        params[PLOT] = False
        params[START_IDS] = start_ids

        params[SVD_METHOD] = svd_method


        if not pool:
            msg = "only pooling (not ww2x) is supported yet for SVDSmoothness, ending"
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
        if not WeightWatcher.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)
     
        #TODO: restrict to ww2x or intra
        layer_iterator = self.make_layer_iterator(model=self.model, layers=layers, params=params)         
        
        # iterate over layers
        #   naive implementation uses just percent, not the actual tail
        #   we eventually want to compute the eigenvectors and analyze them
        #   here we do SVD
        
        for ww_layer in layer_iterator:
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.info("LAYER: {} {}  : {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.framework_layer)))
                
                if method==LAMBDA_MIN:
                    self.apply_esd(ww_layer, params)
                    self.apply_powerlaw(ww_layer, params)
                    params['num_smooth'] = ww_layer.num_pl_spikes
                elif method==DETX:
                    self.apply_esd(ww_layer, params)
                    self.apply_detX(ww_layer, params)
                    params['num_smooth'] = ww_layer.detX_num
                elif method==RMT:
                    self.apply_esd(ww_layer, params)
                    self.apply_mp_fit(ww_layer, random=False, params=params)
                    params['num_smooth'] = ww_layer.num_spikes
                elif method==AUTO:
                    self.apply_esd(ww_layer, params)
                    self.apply_powerlaw(ww_layer, params)
                    alpha = ww_layer.alpha
                    if alpha > 6:
                        self.apply_mp_fit(ww_layer, random=False, params=params)
                        params['num_smooth'] = ww_layer.num_spikes
                    else:
                        self.apply_detX(ww_layer, params)
                        params['num_smooth'] = ww_layer.detX_num   
                    logger.info(f"SVDSmoothing alpha={alpha:0.2f} num_smooth={params['num_smooth']}")
 
                else:
                    params['num_smooth'] = int(percent*ww_layer.M*ww_layer.rf)
                    
                # TODO: do not recompute ESD if present ?
                self.apply_svd_smoothing(ww_layer, params)
        
        logger.info(f"Returning smoothed model ")
        return self.model   

    
    # TODO: add methods that use ETPL-alpha and/or DETX
        
    def apply_svd_smoothing(self, ww_layer, params=None):
        """run truncated SVD on layer weight matrices and reconstruct the weight matrices 
        keep all eigenvlues > percent*ncomp
        if percent < 0, then keep those < than percent*ncomp
        
        Note: can not handle biases yet """
        
        if params is None: params = DEFAULT_PARAMS.copy()
        
        num_smooth = params['num_smooth']
        # BUG HERE
        layer = ww_layer.framework_layer
        layer_id = ww_layer.layer_id
        plot_id = ww_layer.plot_id
        layer_name = ww_layer.name
        layer_type = ww_layer.the_type
        framework = ww_layer.framework
        channels= ww_layer.channels

        
        if framework not in [FRAMEWORK.KERAS, FRAMEWORK.PYTORCH, FRAMEWORK.ONNX, FRAMEWORK.PYSTATEDICT]:
            logger.error("Sorry, SVDSmoothing does not support this model framework ")
            return 

        if channels == CHANNELS.UNKNOWN:
            logger.error("Sorry, SVDSmoothing does not understand the channels for this layer, stopping ")
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
        # TODO fix biases, not working yet
       # old_B = None
        
        logger.info("LAYER TYPE  {} out of {} {} {} ".format(layer_type,LAYER_TYPE.DENSE, LAYER_TYPE.CONV1D, LAYER_TYPE.EMBEDDING))          

        if layer_type in [LAYER_TYPE.DENSE, LAYER_TYPE.CONV1D, LAYER_TYPE.EMBEDDING]:
            if num_smooth > 0:
                logger.debug("Keeping top {} singular values".format(num_smooth))
                new_W = self.smooth_W(old_W, num_smooth, svd_method=params[SVD_METHOD])
            elif num_smooth < 0:
                logger.debug("Chomping off top {} singular values".format(-num_smooth))
                new_W = self.smooth_W_alt(old_W, num_smooth, svd_method=params[SVD_METHOD])
            else:
                logger.warning("Not smoothing {} {}, ncomp=0".format(layer_id, layer_name))
                new_W  = old_W

            new_B = old_B
            # did we flip W when analyzing ?
            if new_W.shape != old_W.shape:
                new_W=new_W.T
                
            self.replace_layer_weights(layer_id, layer, new_W, new_B)

                    
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
                            new_W[i,j,:,:] = self.smooth_W(old_W[i,j,:,:], num_smooth, svd_method=params[SVD_METHOD])
                        elif num_smooth < 0:
                            logger.debug("Chomping off top {} singular values".format(-num_smooth))
                            new_W[i,j,:,:] = self.smooth_W_alt(old_W[i,j,:,:], num_smooth, svd_method=params[SVD_METHOD])
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
                            new_W[:,:,i,j] = self.smooth_W(old_W[:,:,i,j], num_smooth, svd_method=params[SVD_METHOD])
                        elif num_smooth < 0:
                            logger.debug("Chomping off top {} singular values".format(-num_smooth))
                            new_W[:,:,i,j] = self.smooth_W_alt(old_W[:,:,i,j], num_smooth, svd_method=params[SVD_METHOD])
                        else:
                            new_W[:,:,i,j] = old_W[:,:,i,j]
                        
            else:
                logger.warning("Something went wrong, channels not defined or detected for Conv2D layer, layer {} {} skipped ".format(layer_id, layer_name))
            
            self.replace_layer_weights(layer_id, layer, new_W)
    

        else:
            logger.warning("Something went wrong,UNKNOWN layer {} {} skipped , type={}".format(layer_id, layer_name, layer_type))

        return ww_layer
        

    def SVDSharpness(self, model=None,  pool=True, layers=[], plot=False, start_ids=0):
        """Apply the SVD Sharpness Transform to model
        
        layers:
            List of layer ids. If empty, analyze all layers (default)
            If layer ids < 0, then skip the layers specified
            All layer ids must be > 0 or < 0
        
        pool:
            If False, dot not pool the eigenvalues before analyzing
            Use weightwatcher version 0.2x style iterator, which slices up Conv2D layers in N=rf matrices
            
        """
        
        self.set_model_(model)          
         
        params=DEFAULT_PARAMS.copy()
        params[POOL] = pool
        params[LAYERS] = layers
        params[PLOT] = plot
        params[START_IDS] = start_ids

        if not pool:
            msg = "omly pool=True, (not ww2x) is supported yet for SVDSharpness, ending"
            logger.error(msg)
            raise Exception(msg)
        
        # check framework, return error if framework not supported
        # need to access static method on  Model class

        logger.info("params {}".format(params))
        if not WeightWatcher.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        params = self.normalize_params(params)

     
        #TODO: restrict to ww2x or intra
        layer_iterator = self.make_layer_iterator(model=self.model,  layers=layers, params=params)
            
        for ww_layer in layer_iterator:
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.info("LAYER: {} {}  : {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.framework_layer)))
                self.apply_svd_sharpness(ww_layer, params)
        
        logger.info("Returning sharpened model")
        return model  
    
    
    
    def apply_svd_sharpness(self, ww_layer, params=None):
        """run permute layer, run power law, identify and remove the spikes"""
        
        if params is None: params = DEFAULT_PARAMS.copy()
        
        self.apply_permute_W(ww_layer, params)
        self.apply_esd(ww_layer, params)
        self.apply_mp_fit(ww_layer, random=False, params=params)

        params['num_smooth'] = - ww_layer.num_spikes
        logger.debug("Detected {} spikes".format(ww_layer.num_spikes))
        
        self.apply_svd_smoothing(ww_layer, params)
        self.apply_unpermute_W(ww_layer, params)

        return ww_layer


    
    # TODO: put this on the layer itself
    def replace_layer_weights(self, idx, framework_layer, W, B=None):
        """Replace the old layer weights with the new weights in the framework layer"""
    
        framework_layer.replace_layer_weights(W, B=B)
        return


    def analyze_vectors(self, model=None, layers=[], min_evals=DEFAULT_MIN_EVALS, max_evals=DEFAULT_MAX_EVALS,
                plot=True,  savefig=DEF_SAVE_DIR, channels=None):
        """Seperate method to analyze the eigenvectors of each layer"""
        
        self.set_model_(model)          
        
        params=DEFAULT_PARAMS.copy()
        params[SAVEFIG] = savefig
        
        logger.debug("params {}".format(params))
        if not WeightWatcher.valid_params(params):
            msg = "Error, params not valid: \n {}".format(params)
            logger.error(msg)
            raise Exception(msg)
        
        params = self.normalize_params(params)
        logger.info("params {}".format(params))

        layer_iterator = self.make_layer_iterator(model=self.model, layers=layers, params=params)
        
        for id, ww_layer in enumerate(layer_iterator):
            if not ww_layer.skipped and ww_layer.has_weights:
                logger.info("LAYER: {} {}  : {}".format(ww_layer.layer_id, ww_layer.the_type, type(ww_layer.frameword_layer)))
                self.apply_analyze_eigenvectors(ww_layer, params)
                
        return   


    def apply_analyze_eigenvectors(self, ww_layer, params=None):
        """Compute and plot eigenvectors metrics: 

        - vector_entropies
        - localization_ratios
        - participation_ratios
        
        Note:  no  normalization is applied
        
        Does not modify the ww_layer (yet)
        
        """
        
        if params is None: params = DEFAULT_PARAMS.copy()
        
        layer_id = ww_layer.layer_id
        plot_id = ww_layer.plot_id
        name = ww_layer.name or ""
        layer_name = "{} {}".format(plot_id, name)

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

            evals, V = eig_full(X, method=params[SVD_METHOD])
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
            fig, axs = plt.subplots(4, figsize=(8, 12))
            
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
                save_fig(plt, "vector_metrics", ww_layer.plot_id, savedir)
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
                    save_fig(plt, "vector_histograms", ww_layer.plot_id, savedir)
                plt.show(); plt.clf()
       
       
        return
    
    
    # TODO: add
    #   NOT IMPLEMENTED YET BELOW
    # 
    @staticmethod
    def extract_safetensors_config_(weights_dir, model_name, state_dict_filename, start_id = 0):
        # check that filename ends with safetensors
        if not state_dict_filename.endswith(".safetensors"):
            logger.fatal("safetensos file must end with ,safetensors, stopping")
        return WeightWatcher.extract_pytorch_statedict_(weights_dir, model_name, state_dict_filename, start_id = start_id, format=MODEL_FILE_FORMATS.SAFETENSORS, save=False)

    # helper methods for pre-processinf pytorch state_dict files
    @staticmethod
    def extract_pytorch_statedict_(weights_dir, model_name, state_dict_filename, start_id = 0, format=MODEL_FILE_FORMATS.PYTORCH, save=None):
        """Read a pytorch state_dict file, and return a dict of layer configs
        
           Can read model.bin or model.safetensors (detects by filename)
        Parameters:
        
             weights_dir :  ww temp dir with the extracted weights and biases files
             
             model_name: prefix to  the weights files
             
             state_dict_filename: name of the pytorch_model.bin  or safetensors file
             
             start_id: int to start layer id counter
             
             format:  default id MODEL_FILE_FORMATS.PYTORCH, you must set SAFETENSORS to extract it
                          
             save:  (not set by default) save is True for ww_flatfies , False otherwise (use
         
        Returns:
        
            config[layer_id]={name, longname, weightfile, biasfile}
        
        
        Note:  Currently only process dense layers (i.e. transformers), and
               We may not want every layer in the state_dict
               
        Moreover, there is no operational reason to extract safetensors;
             it is only here for testing and may be removed
             that is, to extract the config file
        
        """
        
        config = {}
        
        weight_keys = {}
        if os.path.exists(state_dict_filename):
            logger.debug(f"Reading {state_dict_filename}")
            if format==MODEL_FILE_FORMATS.PYTORCH and state_dict_filename.endswith(".bin"):
                state_dict = torch.load(state_dict_filename, map_location=torch.device('cpu'))
                logger.info(f"Read pytorch state_dict: {state_dict_filename}, len={len(state_dict)}")
                weight_keys = [key for key in state_dict.keys() if 'weight' in key.lower()]
                # we only want the modell but sometimes the state dict has more info
                if 'model' in [str(x) for x in state_dict.keys()]:
                    state_dict = state_dict['model']

            elif format==MODEL_FILE_FORMATS.SAFETENSORS and state_dict_filename.endswith(".safetensors"):

                # since safetensors does not keep the order of the keys we need to
                # read a file with a list keys for the safetensors file
                # 
                #TODO: move this to its own method
                from safetensors import safe_open

                #state_dict = {k: f.get_tensor(k) for k in safe_open(state_dict_filename, framework="pt", device='cpu').keys()}
                state_dict = {}
                with safe_open(state_dict_filename, framework="pt", device='cpu') as f:
                    for k in f.keys():
                        weight_keys[k] = k
        
                logger.info(f"Read safetensors: {state_dict_filename}, len={len(state_dict)}")
         
        else:
            logger.fatal(f"Format: {format } incorrect and /or PyTorch state_dict {state_dict_filename} not found, stopping")
             

            
        
        for layer_id, weight_key in enumerate(weight_keys):
            
            if 'weight' in weight_key or 'bias' in weight_key:
            # TODO: do not save weights by default
                #  and change bias file name depending on safetensors or not
                #
                layer_id_updated = layer_id+start_id  
                name = f"{model_name}.{layer_id_updated}"
                longname = re.sub('.weight$', '', weight_key)
                     
                if format==MODEL_FILE_FORMATS.PYTORCH:   
                    T = state_dict[weight_key]
                elif format==MODEL_FILE_FORMATS.SAFETENSORS:   
                    with safe_open(state_dict_filename, framework="pt", device='cpu') as f:
                        
                        T= f.get_tensor(weight_key)
                
                shape = len(T.shape)  
                #if shape==2:
                W = torch_T_to_np(T)
                
                # TODO: make sure this works with safetensors also
                the_type = WWFlatFile.layer_type_as_str(W)
                    
                # TODO: is this always corret
                has_bias = None
                
                bias_key = re.sub('weight$', 'bias', weight_key)
                if bias_key in state_dict:
                    
                    if format==MODEL_FILE_FORMATS.PYTORCH:   
                        T = state_dict[bias_key]
                    elif format==MODEL_FILE_FORMATS.SAFETENSORS:   
                        with safe_open(state_dict_filename, framework="pt", device='cpu') as f:
                            T= f.get_tensor(bias_key)
                    
                    b = torch_T_to_np(T) 
                    has_bias = True                 
                    
                # TODO: what about Perceptron layers ?
    
                # Save files by default for ww_flatfiles, but only is save=True for safetensores
                if format!=MODEL_FILE_FORMATS.SAFETENSORS or save:
                    weightfile = f"{name}.weight.npy"
                    biasfile = None
    
                    filename = os.path.join(weights_dir,weightfile)
                    logger.debug(f"saving weights to {filename}")
                    np.save(filename, W)
    
                    if has_bias:
                        biasfile = f"{name}.bias.npy"
                        filename = os.path.join(weights_dir,biasfile)
                        logger.debug(f"saving biases to {filename}")
                        np.save(filename, b)
                        
                # safetensors
                elif format==MODEL_FILE_FORMATS.SAFETENSORS:
                    weightfile = state_dict_filename
                    biasfile = None
                    if has_bias:
                        biasfile = state_dict_filename
                        
                else:
                    logger.fatal("Unknown format {format}, stopping")
                    
        
               
                # TODO
                #  add the position id, 0 by default for weights and bias individuallu
                #  allow other, percepton layers, because we need these later
                #  = allow unknown types!
                
                layer_config = {}
                layer_config['key']=weight_key
                layer_config['name']=name
                layer_config['longname']=longname
                layer_config['weightfile']=weightfile
                layer_config['biasfile']=biasfile
                layer_config['type']=the_type
                layer_config['dims']=json.dumps(W.shape)
     
                config[int(layer_id_updated)]=layer_config
                
        return config
    
    
    
    @staticmethod 
    def describe_pytorch_bins(model_dir=None,  **kwargs):
        return WeightWatcher.apply_watcher_to_pytorch_bins(method=METHODS.DESCRIBE, model_dir=model_dir, **kwargs)
    
    @staticmethod 
    def analyze_pytorch_bins(model_dir=None,  **kwargs):
        return WeightWatcher.apply_watcher_to_pytorch_bins(method=METHODS.ANALYZE, model_dir=model_dir, **kwargs)
    
    
    @staticmethod 
    def apply_watcher_to_pytorch_bins(method=describe, model_dir=None, model_name=None, **kwargs):
        """Read the pytorch config and state_dict files, and describe or analyze)
        Notice: the .bin files are parts of oystatedict files, but may or may not contain the ['model'] key
         
        Parameters:  
        
          model_dir:  string, directory of the config file and  pytorch_model.bin file(s)
          
        Returns:
        
          aggregated details dataframe
          
          TODO: maybe intgerate EXTRACT into this
          
          TODO:  this will be depracted soon
        """
        
        total_details = None

        logger.info(f"apply_watcher_to_pytorch_bins files in {model_dir}, returning combined details")    
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            
            try:                  
                # read config, optional
                if model_name is None:
                    config_filename = os.path.join(model_dir, "config.json")
                    with open(config_filename, "r") as f:
                        model_config = json.loads(f.read())
                            
                    model_name = None
                    if 'model_type' in model_config:
                        model_name = model_config['model_type']
                    if model_name is None:
                         model_name = "UNK"
                    
                logger.info(f"Processing model named: {model_name}")
                
                details = None
                
                # read all pytorch bin files, extract all files, and process
                # note: this is not as smart as it could be but better than using all that memory
                # maybe: need to input the glob, or user has to rename them
                # this has never been tested with more than 1 bin file; maybe not necessary
                start_id = 0
                for state_dict_filename in sorted(glob.glob(f"{model_dir}/pytorch_model*bin")):
                    logger.info(f"loading {state_dict_filename}")
                    model = torch.load(state_dict_filename, map_location=torch.device('cpu'))
                    watcher = WeightWatcher(model=model)
                    
                    # set framework
                    
                    if method == METHODS.DESCRIBE:
                        details = watcher.describe(start_ids = start_id, **kwargs)
                    elif method == METHODS.ANALYZE:
                        details = watcher.analyze(start_ids = start_id, **kwargs) 
                        
                                 
                    if len(details) > 0:
                        details['model_name'] = state_dict_filename
                        details['model_filename'] = state_dict_filename
                       
                        if total_details is None:
                            total_details = details
                        else:
                            #details.layer_id  = details.layer_id + start_id
                            total_details = pd.concat([total_details,details], ignore_index=True)      
                              
                    start_id = total_details.layer_id.max()+1

                    logger.info(f"num layer_ids {len(details)} last layer_id {start_id}")
                
                #https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
               
                        
                
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.fatal(f"Unknown problem, stopping")
            
        else:
            logger.fatal(f"Unknown model_dir {model_dir}, stopping")
            
            
            
        return total_details
    
    #
    # TODO: change names
    #
    # ONLY RUN INTENRALLY:
    #
    @staticmethod 
    def extract_pytorch_bins(model_dir=None, model_name=None, tmp_dir="/tmp", format=None):
        """Read the pytorch config and state_dict files, and create tmp direct, and write W and b .npy files
        
        Used currently to evaluate very large models downloaded from HuggingFace 
        
        Notice: the .bin files are parts of pystatedict files, but may or may not contain the ['model'] key
         
        Can read traditional pytorch_model.bin format and huggingface safe_tensors format

        Parameters:  
        
            model_dir:  string, directory of the config file and  pytorch_model.bin file(s)
            
            model_name:  name of model; if None, method attempts to read config.json in the model_dir
            
            tmp_dir:  root directory for the ww tmp weights_dir  that will contain the flat files
            
            format: 'pytorch' | 'safetensors' | None (will infer format it None)
            
        Returns:   a config which has a name, and layer_configs
        
            config = {
            
                model_name: '...'
                weights_dir: /tmp/...
                layers = {0: layer_config, 1: layer_config, 2:layer_config, ...}
            }
            
        Note: we don't support tf_model.h5 files yet, but this could possible be done in the same way
        The only tf_model.h5 I am currently aware of is for BERT
            
        BE CAREFUL using tmp_dir
        """
        
        weights_dir = tempfile.mkdtemp(dir=tmp_dir, prefix="ww_")
        logger.info(f"extract_pytorch files in {model_dir} and placing them in {weights_dir}")
        
        config = {}
        config['framework'] = FRAMEWORK.PYTORCH
        config['weights_dir'] = weights_dir
        
        # TODO:  infer the format ?
        if format is None:
            format, fileglob = WeightWatcher.infer_model_file_format(model_dir)
            logger.info(f"Inferred format, found: {format}")

        config['format'] = format
        if format==MODEL_FILE_FORMATS.PYTORCH:
            fileglob = f"{model_dir}/pytorch_model*bin"      
        elif format==MODEL_FILE_FORMATS.SAFETENSORS:    
            fileglob = f"{model_dir}/model*safetensors"
        else:
            logger.fatal(f"Unknown file format {format}, quitting")
            

        logger.debug(f"searching for files {fileglob}")
    
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            
            try:
                         
                # read config, optional
                if model_name is None:
                    config_filename = os.path.join(model_dir, "config.json")
                    with open(config_filename, "r") as f:
                        model_config = json.loads(f.read())
        
                    model_name = model_config['model_type']
                    if model_name is None:
                        model_name = "UNK"
                    
                config['model_name'] = model_name
                logger.info(f"Processing model named: {model_name}")
                
                config['layers'] = {}
                
                
                # read all pytorch bin files, extract all files, and process
                # note: this is not as smart as it could be but better than using all that memory
                # maybe: need to input the glob, or user has to rename them
                # this has never been tested with more than 1 bin file; maybe not necessary
                start_id = 0
                for state_dict_filename in sorted(glob.glob(fileglob)):
                    logger.info(f"reading and extracting {state_dict_filename}")
                    # TODO:  update layer ids
                    layer_configs = WeightWatcher.extract_pytorch_statedict_(weights_dir, model_name, state_dict_filename, start_id, format=format)
                    start_id = np.max([int(x) for x in layer_configs.keys()]) + 1
                    config['layers'].update(layer_configs) 
                    logger.debug(f"next start_id = {start_id}")
                
                #https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
                config_filename   = os.path.join(weights_dir,WW_CONFIG_FILENAME)
                with open(config_filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)
                        
                
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.fatal(f"Unknown problem, stopping")
            
        else:
            logger.fatal(f"Unknown model_dir {model_dir}, stopping")
            
            
    
    
        return config
       
       
    @staticmethod 
    def write_pystatedict_config(weights_dir,  config):
        """write the config dict to the (tmp) weights dir"""
        
        filename = os.path.join(weights_dir,"ww.config")
        logger.info(f"Writing pystatedict config to {filename} ")
        with open(filename, "w") as f:
            json.dump(config, f)
            
        return filename
    
    @staticmethod 
    def read_pystatedict_config(model_dir):
        """read the pystate config file, ww_config, 
        which has been previously created by running
        
        this method fixes the layer keys to be ints, not strings
        """
        
        # check file is present ?
        filename = os.path.join(model_dir,WW_CONFIG_FILENAME)
        with open(filename, "r") as f:
            config = json.load(f)
            
        # convert keys to strings for comparison
        fixed_config = config
        fixed_config['layers'] = {int(k):v for k,v in config['layers'].items()}
            
            
        return fixed_config
    
    @staticmethod 
    def found_pystate_config(model_dir):
        found = False
        if os.path.isdir(model_dir):
            config_filename = os.path.join(model_dir, 'ww_config')
            found = os.path.isfile(config_filename)
            
        return found
    
    @staticmethod 
    def layer_type_from_str(str_type):
        """convert a string to the layer type enum"""
        
        the_type = LAYER_TYPE.UNKNOWN
        if str_type==NORM:
            the_type = LAYER_TYPE.NORM
        elif str_type==DENSE:
            the_type = LAYER_TYPE.DENSE
        elif str_type==CONV2D:
            the_type = LAYER_TYPE.CONV2D
        elif str_type==CONV1D:
            the_type = LAYER_TYPE.CONV1D
        return the_type


    @staticmethod
    def read_keras_h5_file(model_dir):

        WeightWatcher.load_framework_imports(FRAMEWORK.KERAS)
        
        fileglob = f"{model_dir}/*.h5"
        matching_files = glob.glob(fileglob)
        num_files = len(matching_files)

        if num_files != 1:
            logger.fatal("Sorry, can only read 1 keras h5 file at a time.")
            raise ValueError("Multiple or no Keras H5 files found.")

        model_filename = matching_files[0]  # since num_files must be 1 here
        try:
            logger.debug(f"Reading Keras H5 file {model_filename}")
            loaded_model = tf.keras.models.load_model(model_filename)
            return loaded_model
        except Exception as e:
            logger.fatal(f"Failed to read Keras h5 file {model_filename}. Error: {str(e)}")
            raise  # This will re-raise the caught exception, allowing you to see the full traceback if needed.
        
        

# TODO;
#  get weight and bias .. check how we do it internally

class SafeTensorDict(dict):
    
    """ Stored N safetensors file handles and access them lazily in any order 
    Note: layers are usually named name.weight, name.bias  -- this need to be dealt with"""
    
    import glob
    from safetensors import safe_open
    import logging

    def __init__(self, fileglob, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the dict
        try:
            self.handles = self.open_safetensor_handles(fileglob)  # Open the handles and store them
            logging.info(f"SafetensorDict initialized with files: {fileglob}")

        except Exception as e:
            logging.error(f"Failed to initialize SafetensorDict with files: {fileglob}")
            raise e
        
        
        for handle in self.handles:
            for key in handle.keys():
                self[key] = None  # Populate with keys, values are lazy-loaded
                
        return
    
    
    @staticmethod
    def open_safetensor_handles(fileglob):
        """Read a list of open files"""
        try:
            logging.info(f"Opening safetensor handles for files: {fileglob}")
            handles = []
            for state_dict_filename in sorted(glob.glob(fileglob)):
                logging.info(f"Opening : {state_dict_filename}")
                f = safe_open(state_dict_filename, framework="pt", device='cpu')
                handles.append(f)
            return handles
        except Exception as e:
            logging.error(f"Failed to open safetensor handles for files: {fileglob}")
            raise e
    
    def close(self):
        """Close all handles and clean up"""
        try:
            self.close_safetensor_handles(self.handles)
            self.handles = []
            self.clear()  # Optionally clear the dictionary
            logging.info("SafetensorDict handles closed and dictionary cleared")
        except Exception as e:
            logging.error("Failed to close SafetensorDict handles")
            raise e
    
    @staticmethod
    def close_safetensor_handles(handles):
        """Close all handles in the list"""
        try:
            for f in handles:
                f.close()
            logging.info("Safetensor handles closed successfully")
        except Exception as e:
            logging.error("Failed to close safetensor handles")
            raise e
    
    def get_value(self, key):
        try:
            value = None
            for f in self.handles:
                if key in f.keys():
                    value = f.get_tensor(key)
                    return value
            if value is None:
                logger.info(f"Key {key} not found in safetensors files, returning []")
                return []
                #raise KeyError(f"Key {key} not found in safetensors files.")
        except Exception as e:
            logging.error(f"Error occurred while getting value for key")
            raise e
    
    def __getitem__(self, key):
        return self.get_value(key)
    

        
        
    #
    # def extract_deltas(self, model=None,  base_model=None, layers=[]):
    #     """Extracts the deltas (i.e fine-tuning updates) 
    #
    #       Note: for LoRA updates, this could be a problem since we get back a full rank matrix, 
    #       not the low ank update
    #     """
    #
    #     # TODO:  
    #     #   save model weights in what format ?
    #     # since thats the case, we don't really save anything unless we can recover the W rank
    #     layer_iterator = self.make_layer_iterator(model=self.model, layers=layers, params=params, base_model=base_model)    
    #     for ww_layer in  layer_iterator:
    #         print(ww_layer)
        

