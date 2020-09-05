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

import sys, os 
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import powerlaw
 
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
import torch.nn as nn

#import tensorflow.keras.models.load_model
import pandas as pd

#from .RMT_Util import *
from .RMT_Util import *

#from .constants import *
from .constants import *


MAX_NUM_EVALS= 1000

def main():
    """
    Weight Watcher
    """
    print("WeightWatcher command line support coming later. https://calculationconsulting.com")


class WeightWatcher:

    def __init__(self, model=None, log=True, logger=None):
        self.model = self.load_model(model)
        self.results = {}
        self.summary = {}
        self.logger_set(log=log, logger=logger)

        self.info(self.banner())


    def logger_set(self, log=True, logger=None):
        self.log = log
        self.logger = None
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers: # do not register handlers more than once
                if log:
                    #logging.basicConfig(level=logging.DEBUG)
                    log_level = logging.INFO
                    self.logger.setLevel(log_level)
                    console_handler = logging.StreamHandler()
                    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)
                else:
                    self.logger.addHandler(logging.NullHandler())


    def header(self):
        """WeightWatcher v0.1.dev0 by Calculation Consulting"""
#        from weightwatcher import __name__, __version__, __author__, __description__, __url__
#        return "{} v{} by {}\n{}\n{}".format(__name__, __version__, __author__, __description__, __url__)
        return ""

    def banner(self):
        versions  = "\npython      version {}".format(sys.version)
        versions += "\nnumpy       version {}".format(np.__version__)
        versions += "\ntensforflow version {}".format(tf.__version__)
        versions += "\nkeras       version {}".format(tf.keras.__version__)
        return "\n{}{}".format(self.header(), versions)


    def __repr__(self):
        done = bool(self.results)
        txt  = "\nAnalysis done: {}".format(done)
        return "{}{}".format(self.header(), txt)


    def debug(self, message):
        if self.log:
            self.logger.debug(message)


    def info(self, message):
        if self.log:
            self.logger.info(message)


    def warn(self, message):
        if self.log:
            self.logger.warning(message)


    def error(self, message):
        if self.log:
            self.logger.error(message)


    def load_model(self, model):
        """Load a model from a file if necessary.  Only works for Keras
        """
        res = model
        if isinstance(model, str):
            if os.path.isfile(model):
                self.info("Loading model from file '{}'".format(model))
                res =tf.keras.models.load_model(model)
            else:
                self.error("Loading model from file '{}': file not found".format(model))
        return res


    def model_is_valid(self, model=None):
        model = model or self.model
        if not model:
            return False

        return True


    # test with https://github.com/osmr/imgclsmob/blob/master/README.md
    def analyze(self, model=None, layers=[], min_size=3, max_size=10000,
                alphas=False, lognorms=True, spectralnorms=False, softranks=False,
                normalize=False, glorot_fix=False, plot=False, mp_fit=False, conv2d_fft=False,
                fit_bulk = False):
        """
        Analyze the weight matrices of a model.

        layers:
            List of layer ids. If empty, analyze all layers (default)
        min_size:
            Minimum weight matrix size to analyze
        max_size:
            Maximum weight matrix size to analyze (0 = no limit)
        normalize:
            Normalize the X matrix. Usually True for Keras, False for PyTorch
        glorot_fix:
            Adjust the norm for the Glorot Normalization
        alphas:
            Compute the power laws (alpha) of the weight matrices. 
            Time consuming so disabled by default (use lognorm if you want speed)
        lognorms:
            Compute the log norms of the weight matrices.
        spectralnorms:
            Compute the spectral norm (max eigenvalue) of the weight matrices.
        softranks:
            Compute the soft norm (i.e. StableRank) of the weight matrices.
        mp_fit:
            Compute the best Marchenko-Pastur fit of each weight matrix ESD
        conv2d_fft:
            For Conv2D layers, use FFT method.  Otherwise, extract and combine the weight matrices for each receptive field
            Note:  for conf2d_fft, the ESD is automatically subsampled to max_size eigenvalues max
        fit_bulk: 
            Attempt to fit bulk region of ESD only
        device: N/A yet
            if 'gpu'  use torch.svd()
            else 'cpu' use np.linalg.svd
        """

        model = model or self.model        
        res = {}

        # Treats Custom Conv1D / Attention Layers (ex: GPT, BERT)
        # since they have custom subclass from nn.Module (OpenAIGPTModel)
        def isPyTorchLinearOrConv1D(l):
            tf = False
            if isinstance(l, nn.Conv1d):
                tf = True
            if isinstance(l, nn.Module):
                if hasattr(l, 'weight'):
                    #if isinstance(l, nn.modules.batchnorm._BatchNorm):
                    if "BatchNorm" in str(type(l)):
                        return False
                    w = l.weight.detach().numpy()
#                    tf = True
                    if len(w.shape)==2: # Linear
                        if w.shape[1] >= 2:
                            tf = True
            return tf

        if not isinstance(layers, list):
            layers = [layers]
        layer_ids = [x for x in layers if str(x).isdigit()]
        layer_types = [x for x in layers if isinstance(x, LAYER_TYPE)]

        if not self.model_is_valid(model):
            self.error("Invalid model")
            return res

        if hasattr(model, 'name'):
            # keras has a 'name' attribute on the model
            self.info("Analyzing model '{}' with {} layers".format(model.name, len(model.layers)))
        else:
            # pyTorch has no 'name'
            self.info("Analyzing model")

        weights = []
        
        layers = []
        if hasattr(model, 'layers'):
            # keras
            layers = model.layers
        else:
            # pyTorch
            layers = model.modules()
            
        import torch.nn as nn

        for i, l in enumerate(layers):
            self.debug("Layer {}: {}".format(i+1, l))
            res[i] = {"id": i}
            res[i]["type"] = l
            
            weights = []

            # Filter out layers by numerical id (if any provided)
            if (len(layer_ids) > 0 and (i not in layer_ids)):
                msg = "Skipping (Layer id not requested to analyze)"
                self.debug("Layer {}: {}".format(i+1, msg))
                res[i]["message"] = msg
                continue

            # DENSE layer (Keras) / LINEAR (pytorch)
            #if isinstance(l, keras.layers.core.Dense) or isinstance(l, nn.Linear):
            # TF 2.x
            if isinstance(l, tf.keras.layers.Dense) or isinstance(l, nn.Linear):

                res[i]["layer_type"] = LAYER_TYPE.DENSE

                # Filter out layers by type (if any provided)
                if (len(layer_types) > 0 and
                        not any(layer_type & LAYER_TYPE.DENSE for layer_type in layer_types)):
                    msg = "Skipping (Layer type not requested to analyze)"
                    self.debug("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue
                
                if isinstance(l, nn.Linear):
                    # pyTorch
                    weights = [np.array(l.weight.data.clone().cpu())]
                    receptive_field_size = l.weight.data[0][0].numel()
                else:
                    # keras
                    weights = l.get_weights()[0:1] # keep only the weights and not the bias
#                    weights = l.get_weights()[0:1]  #Tf.Keras.default Glorot uniform
                    
                    # TODO: add option to append bias matrix
                    #if add_bias:
                    #    weights = weigths[0]+weights[1]

                if weights[0].shape[1] < 2:
                    msg = "Skipping (Found array with 1 feature(s) while a minimum of 2 is required)"
                    self.debug("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue
                
                W = weights[0]
                M, N = np.min(W.shape), np.max(W.shape)
                

            # CONV1D layer
            elif (isPyTorchLinearOrConv1D(l)):
                res[i] = {"layer_type": LAYER_TYPE.CONV1D}

                if (len(layer_types) > 0 and
                        not any(layer_type & LAYER_TYPE.CONV1D for layer_type in layer_types)):
                    msg = "Skipping (Layer type not requested to analyze)"
                    self.debug("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue

                weights = [np.array(l.weight.data.clone().cpu())]
                receptive_field_size = l.weight.data[0][0].numel()

                if weights[0].shape[1] < 2:
                    msg = "Skipping (Found array with 1 feature(s) while a minimum of 2 is required)"
                    self.debug("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue
                
                W = weights[0]
                M, N = np.min(W.shape), np.max(W.shape)
            
            elif (isinstance(l, tf.keras.layers.Conv1D)):                
                res[i] = {"layer_type": LAYER_TYPE.CONV1D}

                if (len(layer_types) > 0 and
                        not any(layer_type & LAYER_TYPE.CONV1D for layer_type in layer_types)):
                    msg = "Skipping (Layer type not requested to analyze)"
                    self.debug("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue
                
                weights = l.get_weights()[0:1] # keep only the weights and not the bias
                if weights[0].shape[1] < 2:
                    msg = "Skipping (Found array with 1 feature(s) while a minimum of 2 is required)"
                    self.debug("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue
                
                W = weights[0]
                M, N = np.min(W.shape), np.max(W.shape)
                
            # CONV2D layer
            elif isinstance(l, tf.keras.layers.Conv2D) or isinstance(l, nn.Conv2d):

                res[i] = {"layer_type": LAYER_TYPE.CONV2D}

                if (len(layer_types) > 0 and
                        not any(layer_type & LAYER_TYPE.CONV2D for layer_type in layer_types)):
                    msg = "Skipping (Layer type not requested to analyze)"
                    self.debug("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue
                
                #TODO:  check is this right ?
                if isinstance(l, nn.Conv2d):
                    w = [np.array(l.weight.data.clone().cpu())]
                    receptive_field_size = l.weight.data[0][0].numel()
                else:
                    w = l.get_weights()
                    
                # Run FFT on all channels or just get slices 
                if conv2d_fft:
                    weights, N, M, n_comp = self.get_conv2D_fft(w[0])
                    conv2d_norm = True # ?
                else:
                    weights, N, M = self.get_conv2D_Wmats(w[0])
                    n_comp = M
                    conv2d_norm = True
                
                if weights[0].shape[1] < 2:
                    msg = "Skipping (Found array with 1 feature(s) while a minimum of 2 is required)"
                    self.debug("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue
                
            else:
                msg = "Skipping (Layer not supported)"
                self.debug("Layer {}: {}".format(i+1, msg))
                res[i]["message"] = msg
                continue

            self.debug("Layer {}: Analyzing {} weight matrices...".format(i+1, len(weights)))

            if softranks and not lognorms:
                lognorms = True

            layer_id = i
            ### CHM: I think weights > 0 only for Conv2D layers, but maybe for attention layers too ?
            ### TODO: if combine, then combine all the weights evals, then analyze
            #results = self.analyze_weights(weights, layer_id, min_size, max_size,
            #                               alphas, lognorms, spectralnorms, softranks,
            #                               normalize, glorot_fix, plot, mp_fit)
            layerid = i
            results = self.analyze_combined_weights(weights, layerid, min_size, max_size,
                                                    normalize, glorot_fix, plot, mp_fit, 
                                                    conv2d_norm, N, M, n_comp,
                                                    fit_bulk)

         
            if not results:
                msg = "No weigths to analyze"
                self.debug("Layer {}: {}".format(i+1, msg))
                res["message"] = msg
            else:
                res[i] = {**res[i], **results}
                
        self.results = res
        
        self.print_results(results=res)

        return res
    
    
    def print_results(self, results=None):
        self.compute_details(results=results)

    def get_details(self, results=None):
        """
        Return a pandas dataframe with details for each layer
        """
        df = self.compute_details(results=results)
        details =  df[:-1].dropna(axis=1, how='all').set_index("layer_id") # prune the last line summary
        return details[details.layer_type.notna()]

    def compute_details(self, results=None):
        """
        Return a pandas dataframe with details for each layer
        """
        import numpy as np
        
        if results is None:
            results = self.results

        if not results:
            self.warn("No results to print")
            return

        self.info("### Printing results ###")

        # not all implemented for detais, many are jsut for debugging
        metrics = {
            # key in "results" : pretty print name
            "D": "D",
            "D2": "D2",
            "norm": "Norm",
            "lognorm": "LogNorm",
            "alpha": "Alpha",
            "alpha2": "Alpha2",
            "alpha_weighted": "Alpha Weighted",
            "alpha2_weighted": "Alpha2 Weighted",
            "spectralnorm": "Spectral Norm",
            "logspectralnorm": "Log Spectral Norm",
            "softrank": "Softrank",
            "softranklog": "Softrank Log",
            "softranklogratio": "Softrank Log Ratio",
            "sigma_mp": "Marchenko-Pastur (MP) fit sigma",
            "numofSpikes": "Number of spikes per MP fit",
            "ratio_numofSpikes": "aka, percent_mass, Number of spikes / total number of evals",
            "softrank_mp": "Softrank for MP fit",
            "logpnorm": "alpha pNorm"
        }

        metrics_stats = []
        for metric in metrics:
            metrics_stats.append("{}_min".format(metric))
            metrics_stats.append("{}_max".format(metric))
            metrics_stats.append("{}_avg".format(metric))

            metrics_stats.append("{}_compound_min".format(metric))
            metrics_stats.append("{}_compound_max".format(metric))
            metrics_stats.append("{}_compound_avg".format(metric))

        columns = ["layer_id", "layer_type", "N", "M", "layer_count", "slice", 
                   "slice_count", "level", "comment"] + [*metrics] + metrics_stats
        df = pd.DataFrame(columns=columns)

        metrics_values = {}
        metrics_values_compound = {}

        for metric in metrics:
            metrics_values[metric] = []
            metrics_values_compound[metric] = []

        layer_count = 0
        for layer_id, result in results.items():
            layer_count += 1

            layer_type = np.NAN
            if "layer_type" in result:
                layer_type = str(result["layer_type"]).replace("LAYER_TYPE.", "")

            compounds = {} # temp var
            for metric in metrics:
                compounds[metric] = []

            slice_count = 0
            Ntotal = 0
            Mtotal = 0
            for slice_id, summary in result.items():
                if not str(slice_id).isdigit():
                    continue

                slice_count += 1

                N = np.NAN
                if "N" in summary:
                    N = summary["N"]
                    Ntotal += N

                M = np.NAN
                if "M" in summary:
                    M = summary["M"]
                    Mtotal += M

                data = {"layer_id": layer_id, "layer_type": layer_type, "N": N, "M": M, "slice": slice_id, "level": LEVEL.SLICE, "comment": "Slice level"}
                for metric in metrics:
                    if metric in summary:
                        value = summary[metric]
                        if value is not None:
                            metrics_values[metric].append(value)
                            compounds[metric].append(value)
                            data[metric] = value
                row = pd.DataFrame(columns=columns, data=data, index=[0])
                df = pd.concat([df, row])

            data = {"layer_id": layer_id, "layer_type": layer_type, "N": Ntotal, "M": Mtotal, "slice_count": slice_count, "level": LEVEL.LAYER, "comment": "Layer level"}
            # Compute the coumpound value over the slices
            for metric, value in compounds.items():
                count = len(value)
                if count == 0:
                    continue

                compound = np.mean(value)
                metrics_values_compound[metric].append(compound)
                data[metric] = compound

                if count > 1:
                    # Compound value of the multiple slices (conv2D)
                    self.debug("Layer {}: {} compound: {}".format(layer_id, metrics[metric], compound))
                else:
                    # No slices (Dense or Conv1D)
                    self.debug("Layer {}: {}: {}".format(layer_id, metrics[metric], compound))

            row = pd.DataFrame(columns=columns, data=data, index=[0])
            df = pd.concat([df, row])

        data = {"layer_count": layer_count, "level": LEVEL.NETWORK, "comment": "Network Level"}
        for metric, metric_name in metrics.items():
            if metric not in metrics_values or len(metrics_values[metric]) == 0:
                continue

            values = metrics_values[metric]
            minimum = min(values)
            maximum = max(values)
            avg = np.mean(values)
            self.summary[metric] = avg
            self.info("{}: min: {}, max: {}, avg: {}".format(metric_name, minimum, maximum, avg))
            data["{}_min".format(metric)] = minimum
            data["{}_max".format(metric)] = maximum
            data["{}_avg".format(metric)] = avg

            values = metrics_values_compound[metric]
            minimum = min(values)
            maximum = max(values)
            avg = np.mean(values)
            self.summary["{}_compound".format(metric)] = avg
            self.info("{} compound: min: {}, max: {}, avg: {}".format(metric_name, minimum, maximum, avg))
            data["{}_compound_min".format(metric)] = minimum
            data["{}_compound_max".format(metric)] = maximum
            data["{}_compound_avg".format(metric)] = avg

        row = pd.DataFrame(columns=columns, data=data, index=[0])
        df = pd.concat([df, row])
        df['slice'] += 1 #fix the issue that slice starts from 0 and don't match the plot

        return df.dropna(axis=1,how='all')

    
    def get_summary(self, pandas=False):
        if pandas:
            return pd.DataFrame(data=self.summary, index=[0])
        else:
            return self.summary


    def get_conv2D_Wmats(self, Wtensor):
        """Extract W slices from a 4 index conv2D tensor of shape: (N,M,i,j) or (M,N,i,j).  
        Return ij (N x M) matrices
        
        """
        
        self.info("get_conv2D_Wmats")

        Wmats = []
        s = Wtensor.shape
        N, M, imax, jmax = s[0],s[1],s[2],s[3]
        if N + M >= imax + jmax:
            self.debug("Pytorch tensor shape detected: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))
            
            for i in range(imax):
                for j in range(jmax):
                    W = Wtensor[:,:,i,j]
                    if N < M:
                        W = W.T
                    Wmats.append(W)
        else:
            N, M, imax, jmax = imax, jmax, N, M          
            self.debug("Tf.Keras.tensor shape detected: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))
            
            for i in range(imax):
                for j in range(jmax):
                    W = Wtensor[i,j,:,:]
                    if N < M:
                        W = W.T
                    Wmats.append(W)
                    
        self.info("get_conv2D_Wmats N={} M={}".format(N,M))

            
        return Wmats, N, M
    
    
    def get_conv2D_fft(self, W):
        """Compute FFT of Conv2D channels, to apply SVD later"""
        
        self.info("get_conv2D_fft on W {}".format(W.shape))

        # is pytorch or tensor style 
        s = W.shape
        self.debug("    Conv2D SVD ({}): Analyzing ...".format(s))

        N, M, imax, jmax = s[0],s[1],s[2],s[3]
        # probably better just to check what col N is in 
        if N + M >= imax + jmax:
            self.debug("[2,3] tensor shape detected: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))    
            fft_axes = [2,3]
        else:
            N, M, imax, jmax = imax, jmax, N, M          
            fft_axes = [0,1]
            self.debug("[1,2] tensor shape detected: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))

        #  receptive_field / kernel size
        rf = np.min([imax, jmax])
        # aspect ratio
        Q = N/M 
        # num non-zero eigenvalues  rf is receptive field size (sorry calculated again here)
        n_comp = rf*N*M
        
        self.info("N={} M={} n_comp {} ".format(N,M,n_comp))

        # run FFT on each channel
        fft_grid = [32,32]
        fft_coefs = np.fft.fft2(W, fft_grid, axes=fft_axes)
        
        return [fft_coefs], N, M, n_comp



    def normalize_evals(self, evals, N, M):
        """DEPRECATED: Normalize the eigenvalues W by N and receptive field size (if needed)"""
        self.debug(" normalzing evals, N, M {},{},{}".format(N,M))
        return evals/np.sqrt(N)

    def glorot_norm_fix(self, W, N, M, rf_size):
        """Apply Glorot Normalization Fix"""

        kappa = np.sqrt( 2 / ((N + M)*rf_size) )
        W = W/kappa
        return W 

    def pytorch_norm_fix(self, W, N, M, rf_size):
        """Apply pytorch Channel Normalization Fix

        see: https://chsasank.github.io/vision/_modules/torchvision/models/vgg.html
        """

        kappa = np.sqrt( 2/(N*rf_size) )
        W = W/kappa
        return W 


    def glorot_norm_check(self, W, N, M, rf_size, 
                   lower = 0.5, upper = 1.5):
        """Check if this layer needs Glorot Normalization Fix"""

        kappa = np.sqrt( 2 / ((N + M)*rf_size) )
        norm = np.linalg.norm(W)

        check1 = norm / np.sqrt(N*M)
        check2 = norm / (kappa*np.sqrt(N*M))
        
        if (rf_size > 1) and (check2 > lower) and (check2 < upper):   
            return check2, True
        elif (check1 > lower) & (check1 < upper): 
            return check1, True
        else:
            if rf_size > 1:
                return check2, False
            else:
                return check1, False
            
    # make this a static method ?        
    def combined_eigenvalues(self, weights, n_comp, min_size=1, max_size=10000, 
                             normalize=True, glorot_fix=False, conv2d_norm=True):
        """Compute the eigenvalues for all weights of the NxM weight matrices (N >= M), 
            combined into a single, sorted, numpy array
    
            Skips matrices where M < min_size or M > max_size
            Applied normalization and glorot_fix if specified
            
            Assumes an array of weights comes from a conv2D layer and applies conv2d_norm normalization by default
          
            Also returns max singular value and rank_loss, needed for other calculations
         """
         
        all_evals = []
        max_sv = 0.0
        rank_loss = 0

        count = len(weights)
        for  W in weights:
            M, N = np.min(W.shape), np.max(W.shape)
            if M >= min_size:# and M <= max_size:

                Q=N/M
                check, checkTF = self.glorot_norm_check(W, N, M, count) 
    
                # assume receptive field size is count
                if glorot_fix:
                    W = self.glorot_norm_fix(W, N, M, count)
                elif conv2d_norm:
                    # probably never needed since we always fix for glorot
                    W = W * np.sqrt(count/2.0) 
                
                # SVD can be swapped out here
                # svd = TruncatedSVD(n_components=M-1, n_iter=7, random_state=10)

                W = W.astype(float)
                self.info("Running full SVD:  W.shape={}  n_comp = {}".format(W.shape, n_comp))
                sv = np.linalg.svd(W, compute_uv=False)
                sv = sv.flatten()
                sv = np.sort(sv)[-n_comp:]
                if len(sv) > max_size:
                    self.info("chosing {} singular values from {} ".format(max_size, len(sv)))
                    sv = np.random.choice(sv, size=max_size)
                    
                #sv = svd.singular_values_
                evals = sv*sv
                if normalize:
                    evals = evals/N
                 
                all_evals.extend(evals)
                                           
                max_sv = np.max([max_sv, np.max(sv)])
                max_ev = np.max(evals)
                rank_loss = 0#rank_loss + self.calc_rank_loss(sv, M, max_ev)
            
        return np.sort(np.array(all_evals)), max_sv, rank_loss
    
    
    def random_eigenvalues(self, weights, n_comp, num_replicas=1, min_size=1, max_size=10000, 
                           normalize=True, glorot_fix=False, conv2d_norm=True):
        """Compute the eigenvalues for all weights of the NxM Randomized weight matrices (N >= M), 
            combined into a single, sorted, numpy array
    
        see: combined_eigenvalues()
        
         """
         
         
        all_evals = []

        for num in range(num_replicas):
            count = len(weights)
            for  W in weights:
    
                M, N = np.min(W.shape), np.max(W.shape)
                if M >= min_size:# and M <= max_size:
    
                    Q=N/M
                    check, checkTF = self.glorot_norm_check(W, N, M, count) 
        
                    # assume receptive field size is count
                    if glorot_fix:
                        W = self.glorot_norm_fix(W, N, M, count)
                    elif conv2d_norm:
                        # probably never needed since we always fix for glorot
                        W = W * np.sqrt(count/2.0) 
                    
                    Wrand = W.flatten()
                    np.random.shuffle(Wrand)
                    W = Wrand.reshape(W.shape)
           
                    W = W.astype(float)
                    self.info("Running Randomized Full SVD")
                    sv = np.linalg.svd(W, compute_uv=False)
                    sv = sv.flatten()
                    sv = np.sort(sv)[-n_comp:]    
                    
                    #sv = svd.singular_values_
                    evals = sv*sv
                    if normalize:
                        evals = evals/N
                     
                    all_evals.extend(evals)
                                        
            
        return np.sort(np.array(all_evals))
    

    
    
    
    def analyze_combined_weights(self, weights, layerid, min_size, max_size,
                normalize, glorot_fix, plot, mp_fit,  conv2d_norm, N, M, n_comp, 
                fit_bulk):
        """Analyzes weight matrices, combined as if they are 1 giant matrices
        Computes PL alpha fits and  various norm metrics
         - alpha
         - alpha_weighted
         
         - Frobenius norm 
         - Spectral Norm
         - p-norm / Shatten norm
         - Soft Rank / Stable Rank
        
        Assumes all matrices have the same shape, (N x M), N > M.
        
        For now, retains the old idea that we have layer_id and slice_id=0 (always)
        res[0][''] = ...
         """
         
        res = {}
        count = len(weights)
        if count == 0:
            return res

        # slice_id
        i = 0
        res[i] = {}
        
        # TODO:  add conv2D ?  How to integrate into this code base ?
        # how deal with glorot norm and normalization ?
        # what is Q ?  n_comps x something ?
        
        # assume all weight matrices have the same shape
        W = weights[0]
        #M, N = np.min(W.shape), np.max(W.shape)
        Q=N/M
        
        res[i]["N"] = N
        res[i]["M"] = M
        res[i]["Q"] = Q  
        summary = []
         
        # TODO:  start method here, have a pre-method that creates the matrices of weights
        # pass N, M in 
        
        #
        # Get combined eigenvalues for all weight matrices, using SVD
        # returns singular values to
        #
 
        check, checkTF = self.glorot_norm_check(W, N, M, count) 
        res[i]['check'] = check
        res[i]['checkTF'] = checkTF
        
        
        evals, sv_max, rank_loss = self.combined_eigenvalues(weights, n_comp, min_size, max_size, normalize, glorot_fix, conv2d_norm)  
        
        num_evals = len(evals)     
        if num_evals < min_size:
            self.info("skipping layer, num evals {} < {} min size".format(num_evals, min_size))
            return res
        #elif num_evals > max_size:
        #    self.info("skipping layer, num evals {} > {} max size".format(num_evals, max_size))
        #    return res
        
        
        lambda_max = np.max(evals)
        
        res[i]["sv_max"] = sv_max
        res[i]["rank_loss"] = rank_loss
        
        # this should never happen, but just in case
        if len(evals) < 2: 
            return res
        
        #
        # Power law fit
        #
        title = "Weight matrix ({}x{})  layer ID: {}".format(N, M, layerid)
        alpha, D, xmin, xmax = self.fit_powerlaw(evals, plot=plot, title=title)    
        
        res[i]["alpha"] = alpha
        res[i]["D"] = D
        res[i]["xmin"] = xmin
        res[i]["xmax"] = xmax
        res[i]["lambda_min"] = np.min(evals)
        res[i]["lambda_max"] =lambda_max
        
        
        alpha_weighted = alpha * np.log10(lambda_max)
        res[i]["alpha_weighted"] = alpha_weighted
        
        #
        # other metrics
        #
                  
        norm = np.sum(evals)
        res[i]["norm"] = norm
        lognorm = np.log10(norm)
        res[i]["lognorm"] = lognorm
        
        logpnorm = np.log10(np.sum([ev**alpha for ev in evals]))
        res[i]["logpnorm"] = logpnorm
            
        res[i]["spectralnorm"] = lambda_max
        res[i]["logspectralnorm"] = np.log10(lambda_max)

        summary.append("Weight matrix  ({},{}): LogNorm: {} ".format( M, N, lognorm) )
                
        softrank = norm**2 / sv_max**2
        softranklog = np.log10(softrank)
        softranklogratio = lognorm / np.log10(sv_max)
        res[i]["softrank"] = softrank
        res[i]["softranklog"] = softranklog
        res[i]["softranklogratio"] = softranklogratio
        
        summary += "{}. Softrank: {}. Softrank log: {}. Softrank log ratio: {}".format(summary, softrank, softranklog, softranklogratio)
        res[i]["summary"] = "\n".join(summary)
        for line in summary:
            self.debug("    {}".format(line))
            
        # overlay plot with randomized matrix on log scale
        num_replicas = 1
        if len(evals) < 100: 
            num_replicas = 10
            
        rand_evals = self.random_eigenvalues(weights, n_comp, num_replicas=num_replicas, 
                                              min_size=min_size, max_size=max_size, 
                                              normalize=normalize, glorot_fix=glorot_fix, conv2d_norm=conv2d_norm)

        if plot:
            self.plot_random_esd(evals, rand_evals, title)       
        
        # power law fit, with xmax = random bulk edge
        # experimental fit
        #
        # note: this only works if we have more than a few eigenvalues < xmax and > xmin
        alpha2, D2, xmin2, xmax2 = None, None, None, None
        if fit_bulk:
            self.info("fitting bulk")
            try:
                xmax = np.max(rand_evals)
                num_evals_left = len(evals[evals < xmax])
                if  num_evals_left > 10: # not sure on this yet
                    title = "Weight matrix ({}x{})  layer ID: {} Fit2".format(N, M, layerid)
                    #alpha2, D2, xmin2, xmax2 = self.fit_powerlaw(evals, xmin='peak', xmax=xmax, plot=plot, title=title) 
                    alpha2, D2, xmin2, xmax2 = self.fit_powerlaw(evals, xmin=None, xmax=xmax, plot=plot, title=title)  
     
                    res[i]["alpha2"] = alpha2
                    res[i]["D2"] = D2
                    alpha2_weighted = alpha2 * np.log10(xmax)
                    res[i]["alpha2_weighted"] = alpha2_weighted
            except:
                self.info("fit2 fails, not sure why")
                pass  
                
        
        return res
            
    def plot_random_esd(self, evals, rand_evals, title):
        """Plot histogram and log histogram of ESD and randomized ESD"""
          
        nonzero_evals = evals[evals > 0.0]
        nonzero_rand_evals = rand_evals[rand_evals > 0.0]
        max_rand_eval = np.max(rand_evals)

        plt.hist((nonzero_evals),bins=100, density=True, color='g', label='original')
        plt.hist((nonzero_rand_evals),bins=100, density=True, color='r', label='random', alpha=0.5)
        plt.axvline(x=(max_rand_eval), color='orange', label='max rand')
        plt.title(r"ESD and Randomized (ESD $\rho(\lambda)$)" + "\nfor {} ".format(title))                  
        plt.legend()
        plt.show()

        plt.hist(np.log10(nonzero_evals),bins=100, density=True, color='g', label='original')
        plt.hist(np.log10(nonzero_rand_evals),bins=100, density=True, color='r', label='random', alpha=0.5)
        plt.axvline(x=np.log10(max_rand_eval), color='orange', label='max rand')
        plt.title(r"Log10 ESD and Randomized (ESD $\rho(\lambda))$" + "\nfor {} ".format(title))                  
        plt.legend()
        plt.show()
        
        
          
        
    # Mmybe should be static function    
    def calc_rank_loss(self, singular_values, M, lambda_max):
        """compute the rank loss for these singular given the tolerances
        """
        sv = singular_values
        tolerance = lambda_max * M * np.finfo(np.max(sv)).eps
        return np.count_nonzero(sv > tolerance, axis=-1)
        
            
    def fit_powerlaw(self, evals, xmin=None, xmax=None, plot=True, title="", sample=True):
        """Fit eigenvalues to powerlaw
        
            if xmin is 
                'auto' or None, , automatically set this with powerlaw method
                'peak' , try to set by finding the peak of the ESD on a log scale
            
            if xmax is 'auto' or None, xmax = np.max(evals)
                     
         """
        
        self.info("fitting power law on {} eigenvalues".format(len(evals)))
        alpha, D =  None, None      
        
        if  sample and len(evals) > MAX_NUM_EVALS:
            self.info("chosing {} eigenvalues from {} ".format(MAX_NUM_EVALS, len(evals)))
            evals = np.random.choice(evals, size=MAX_NUM_EVALS)
                    
        
        if xmax=='auto' or xmax is None:
            xmax = np.max(evals)
            
        if xmin=='auto' or xmin is None:
            fit = powerlaw.Fit(evals, xmax=xmax, verbose=False)
        elif xmin=='peak':
            nz_evals = evals[evals > 0.0]
            num_bins = 100# np.min([100, len(nz_evals)])
            h = np.histogram(np.log10(nz_evals),bins=num_bins)
            ih = np.argmax(h[0])
            xmin2 = 10**h[1][ih]
            xmin_range = (0.95*xmin2, 1.05*xmin2)
            fit = powerlaw.Fit(evals, xmin=xmin_range, xmax=xmax, verbose=False)   
        else:
            fit = powerlaw.Fit(evals, xmin=xmin, xmax=xmax, verbose=False)
            
            
        alpha = fit.alpha 
        D = fit.D
        xmin = fit.xmin
        xmax = fit.xmax
        
  
        if plot:
            fig2 = fit.plot_pdf(color='b', linewidth=2)
            fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig2)
            fit.plot_ccdf(color='r', linewidth=2, ax=fig2)
            fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig2)
        
            title = "Power law fit for {}\n".format(title) 
            title = title + r"$\alpha$={0:.3f}; ".format(alpha) + r"KS_distance={0:.3f}".format(D) +"\n"
            plt.title(title)
            plt.show()
    
            # plot eigenvalue histogram
            num_bins = 100#np.min([100,len(evals)])
            plt.hist(evals, bins=num_bins, density=True)
            plt.title(r"ESD (Empirical Spectral Density) $\rho(\lambda)$" + "\nfor {} ".format(title))                  
            plt.axvline(x=fit.xmin, color='red', label='xmin')
            plt.legend()
            plt.show()


            # plot log eigenvalue histogram
            nonzero_evals = evals[evals > 0.0]
            plt.hist(np.log10(nonzero_evals),bins=100, density=True)
            plt.title(r"Log10 ESD (Empirical Spectral Density) $\rho(\lambda)$" + "\nfor {} ".format(title))                  
            plt.axvline(x=np.log10(fit.xmin), color='red')
            plt.axvline(x=np.log10(fit.xmax), color='orange', label='xmax')
            plt.legend()
            plt.show()
    
            # plot xmins vs D
            plt.plot(fit.xmins, fit.Ds, label=r'$D$')
            plt.axvline(x=fit.xmin, color='red', label='xmin')
            plt.xlabel(r'$x_{min}$')
            plt.ylabel(r'$D,\sigma,\alpha$')
            plt.title("current xmin={:0.3}".format(fit.xmin))
            plt.legend()
            plt.show()   
            
        ### TODOL  find best fit     
    
        return alpha , D, xmin, xmax
        
         

    def analyze_weights(self, weights, layerid, min_size, max_size,
                        alphas, lognorms, spectralnorms, softranks,  
                        normalize, glorot_fix, plot, mp_fit):
        """Analyzes weight matrices.
        
        Example in Tf.Keras.
            weights = layer.get_weights()
            analyze_weights(weights)
        """

        res = {}
        count = len(weights)
        if count == 0:
            return res


        for i, W in enumerate(weights):
            res[i] = {}
            M, N = np.min(W.shape), np.max(W.shape)
            Q=N/M
            res[i]["N"] = N
            res[i]["M"] = M
            res[i]["Q"] = Q
            lambda0 = None

            check, checkTF = self.glorot_norm_check(W, N, M, count) 
            res[i]['check'] = check
            res[i]['checkTF'] = checkTF
            # assume receptive field size is count
            if glorot_fix:
                W = self.glorot_norm_fix(W, N, M, count)
            else:
                # probably never needed since we always fix for glorot
                W = W * np.sqrt(count/2.0) 


            if spectralnorms: #spectralnorm is the max eigenvalues
                
                svd = TruncatedSVD(n_components=1, n_iter=7, random_state=10)
                svd.fit(W)
                sv = svd.singular_values_
                sv_max = np.max(sv)
                evals = sv*sv
                if normalize:
                    evals = evals/N

                lambda0 = np.max(evals)
                res[i]["spectralnorm"] = lambda0
                res[i]["logspectralnorm"] = np.log10(lambda0)

            if M < min_size:
                summary = "Weight matrix {}/{} ({},{}): Skipping: too small (<{})".format(i+1, count, M, N, min_size)
                res[i]["summary"] = summary 
                self.debug("    {}".format(summary))
                continue

            #if max_size > 0 and M > max_size:
            #    summary = "Weight matrix {}/{} ({},{}): Skipping: too big (testing) (>{})".format(i+1, count, M, N, max_size)
            #    res[i]["summary"] = summary 
            #    self.info("    {}".format(summary))
            #    continue

            summary = []
                
            self.debug("    Weight matrix {}/{} ({},{}): Analyzing ..."
                     .format(i+1, count, M, N))
            
            if alphas:

                svd = TruncatedSVD(n_components=M-1, n_iter=7, random_state=10)
                
                try:
                    svd.fit(W) 
                except:
                    W = W.astype(float)
                    svd.fit(W)
                    
                sv = svd.singular_values_
                sv_max = np.max(sv)
                evals = sv*sv
                if normalize:
                    evals = evals/N

                # Other (slower) way of computing the eigen values:
                # X = np.dot(W.T,W)/N
                #evals2 = np.linalg.eigvals(X)
                #res[i]["lambda_max2"] = np.max(evals2)

                #TODO:  add alpha2

                lambda_max = np.max(evals)
                fit = powerlaw.Fit(evals, xmax=lambda_max, verbose=False)
                alpha = fit.alpha 
                D = fit.D
                xmin = fit.xmin
                res[i]["alpha"] = alpha
                res[i]["D"] = D
                res[i]["xmin"] = xmin
                res[i]["lambda_min"] = np.min(evals)
                res[i]["lambda_max"] = lambda_max
                alpha_weighted = alpha * np.log10(lambda_max)
                res[i]["alpha_weighted"] = alpha_weighted
                tolerance = lambda_max * M * np.finfo(np.max(sv)).eps
                res[i]["rank_loss"] = np.count_nonzero(sv > tolerance, axis=-1)
                
                logpnorm = np.log10(np.sum([ev**alpha for ev in evals]))
                res[i]["logpnorm"] = logpnorm

                nz_evals = evals[evals > 0.0]
                num_bins = np.min([100, len(nz_evals)])
                h = np.histogram(np.log10(nz_evals),bins=num_bins)
                ih = np.argmax(h[0])
                xmin2 = 10**h[1][ih]
                if xmin2 > xmin:
                    self.info("resseting xmin2 to xmin")
                    xmin2 = xmin

                fit2 = powerlaw.Fit(evals, xmin=xmin2, xmax=lambda_max, verbose=False)
                alpha2 = fit2.alpha
                D2 = fit2.D
                res[i]["alpha2"] = alpha2
                res[i]["D2"] = D2
                res[i]["xmin2"] = fit2.xmin
                res[i]["alpha2_weighted"] =  alpha2 * np.log10(lambda_max)

                summary.append("Weight matrix {}/{} ({},{}): Alpha: {}, Alpha Weighted: {}, D: {}, pNorm {}".format(i+1, count, M, N, alpha, alpha_weighted, D, logpnorm))

                #if alpha < alpha_min or alpha > alpha_max:
                #    message = "Weight matrix {}/{} ({},{}): Alpha {} is in the danger zone ({},{})".format(i+1, count, M, N, alpha, alpha_min, alpha_max)
                #    self.debug("    {}".format(message))

                if plot:
                    fig2 = fit.plot_pdf(color='b', linewidth=2)
                    fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig2)
                    fit.plot_ccdf(color='r', linewidth=2, ax=fig2)
                    fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig2)
                    fit2.plot_pdf(color='g', linewidth=2)
#                    plt.title("Power law fit for Weight matrix {}/{} (layer ID: {})".format(i+1, count, layerid))
                    title = "Power law fit for Weight matrix {}/{} (layer ID: {})\n".format(i+1, count, layerid) 
                    title = title + r"$\alpha$={0:.3f}; ".format(alpha) + r"KS_distance={0:.3f}".format(D) +"\n"
                    title = title + r"$\alpha2$={0:.3f}; ".format(alpha2) + r"KS_distance={0:.3f}".format(D2)
                    plt.title(title)
                    plt.show()

                    # plot eigenvalue histogram
                    plt.hist(evals, bins=100, density=True)
#                    plt.title(r"ESD (Empirical Spectral Density) $\rho(\lambda)$" + " for Weight matrix {}/{} (layer ID: {})".format(i+1, count, layerid))
                    plt.title(r"ESD (Empirical Spectral Density) $\rho(\lambda)$" + "\nfor Weight matrix ({}x{}) {}/{} (layer ID: {})".format(N, M, i+1, count, layerid))                    
                    plt.axvline(x=fit.xmin, color='red')
                    plt.axvline(x=fit2.xmin, color='green')
                    plt.show()

                    nonzero_evals = evals[evals > 0.0]
                    plt.hist(np.log10(nonzero_evals),bins=100, density=True)
#                    plt.title("Eigen Values for Weight matrix {}/{} (layer ID: {})".format(i+1, count, layerid))
                    plt.title("Logscaling Plot of Eigenvalues\nfor Weight matrix ({}X{}) {}/{} (layer ID: {})".format(N, M, i+1, count, layerid))
                    plt.axvline(x=np.log10(fit.xmin), color='red')
                    plt.axvline(x=np.log10(xmin2), color='green')
                    plt.show()
                
            if mp_fit:
#                if Q == 1:
#                    ## Quarter-Circle Law
#                    sv = svd.singular_values_
#                    to_plot = np.sqrt(sv*sv/N)
#                else:
#                    to_plot = sv*sv/N
#                w_unnorm = W*np.sqrt(N + M)/np.sqrt(2*N)
                
                if not alphas:
                    #W = self.normalize(W, N, M, count)
                    svd = TruncatedSVD(n_components=M-1, n_iter=7, random_state=10)
                    svd.fit(W) 
                    sv = svd.singular_values_
                    sv_max = np.max(sv)
                    evals = sv*sv
                    if normalize:
                        evals = evals/N
                    lambda_max = np.max(evals)
                
                to_plot = evals.copy()
                
                bw = 0.1
#                s1, f1 = RMT_Util.fit_mp(to_plot, Q, bw = 0.01)  
#                s1, f1 = fit_density(to_plot, Q, bw = bw)  
                s1, f1 = fit_density_with_range(to_plot, Q, bw = bw)
                
                res[i]['sigma_mp'] = s1
                
                bulk_edge = (s1 * (1 + 1/np.sqrt(Q)))**2
                
                spikes = sum(to_plot > bulk_edge)
                res[i]['numofSpikes'] = spikes
                res[i]['ratio_numofSpikes'] = spikes / (M - 1)
                
                softrank_mp = bulk_edge / lambda_max
                res[i]['softrank_mp'] = softrank_mp
                
                if plot:
                    
                    if Q == 1:
                        fit_law = 'QC SSD'
                        
                        #Even if the quarter circle applies, still plot the MP_fit
                        plot_density(to_plot, s1, Q, method = "MP")
                        plt.legend([r'$\rho_{emp}(\lambda)$', 'MP fit'])
                        plt.title("MP ESD, sigma auto-fit for Weight matrix {}/{} (layer ID: {})\nsigma_fit = {}, softrank_mp = {}".format(i+1, count, layerid, round(s1, 6), round(softrank_mp, 3)))
                        plt.show()
                        
                    else:
                        fit_law = 'MP ESD'
#                        RMT_Util.plot_ESD_and_fit(model=None, eigenvalues=to_plot, 
#                                                  Q=Q, num_spikes=0, sigma=s1)
                    plot_density_and_fit(model=None, eigenvalues=to_plot, 
                                         Q=Q, num_spikes=0, sigma=s1, verbose = False)
                    plt.title("{}, sigma auto-fit for Weight matrix {}/{} (layer ID: {})\nsigma_fit = {}, softrank_mp = {}".format(fit_law, i+1, count, layerid, round(s1, 6), round(softrank_mp, 3)))
                    plt.show()
                        
            if lognorms:
                norm = np.linalg.norm(W) #Frobenius Norm
                res[i]["norm"] = norm
                lognorm = np.log10(norm)
                res[i]["lognorm"] = lognorm

                X = np.dot(W.T,W)                
                if normalize:
                    X = X/N
                normX = np.linalg.norm(X) #Frobenius Norm
                res[i]["normX"] = normX
                lognormX = np.log10(normX)
                res[i]["lognormX"] = lognormX

                summary.append("Weight matrix {}/{} ({},{}): LogNorm: {} ; LogNormX: {}".format(i+1, count, M, N, lognorm, lognormX))
                
                if softranks: 
                    softrank = norm**2 / sv_max**2
                    softranklog = np.log10(softrank)
                    softranklogratio = lognorm / np.log10(sv_max)
                    res[i]["softrank"] = softrank
                    res[i]["softranklog"] = softranklog
                    res[i]["softranklogratio"] = softranklogratio
                    summary += "{}. Softrank: {}. Softrank log: {}. Softrank log ratio: {}".format(summary, softrank, softranklog, softranklogratio)

                        

            res[i]["summary"] = "\n".join(summary)
            for line in summary:
                self.debug("    {}".format(line))

        return res
    
    
    
