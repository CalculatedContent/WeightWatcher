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

import sys
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import powerlaw
        
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import load_model

from .constants import *


def main():
    """
    Weight Watcher
    """
    print("WeightWatcher command line support coming later. https://calculationconsulting.com")


class WeightWatcher:

    def __init__(self, model=None, log=True, logger=None):
        self.model = self.load_model(model)
        self.alphas = {}
        self.results = {}
        self.summary = {}
        self.logger_set(log=log, logger=logger)

        self.info(self.banner())

    def logger_set(self, log=True, logger=None):
        self.logger = None
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers: # do not register handlers more than once
                if log:
                    #logging.basicConfig(level=logging.DEBUG)
                    log_level = logging.DEBUG
                    self.logger.setLevel(log_level)
                    console_handler = logging.StreamHandler()
                    console_handler.setLevel(log_level)
                    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)
                else:
                    self.logger.addHandler(logging.NullHandler())


    def header(self):
        """WeightWatcher v0.1.dev0 by Calculation Consulting"""
        from weightwatcher import __name__, __version__, __author__, __description__, __url__
        return "{} v{} by {}\n{}\n{}".format(__name__, __version__, __author__, __description__, __url__)


    def banner(self):
        versions  = "\npython      version {}".format(sys.version)
        versions += "\nnumpy       version {}".format(np.__version__)
        versions += "\ntensforflow version {}".format(tf.__version__)
        versions += "\nkeras       version {}".format(keras.__version__)
        return "\n{}{}".format(self.header(), versions)


    def __repr__(self):
        done = bool(self.results)
        txt  = "\nAnalysis done: {}".format(done)
        return "{}{}".format(self.header(), txt)


    def debug(self, message):
        self.logger.debug(message)


    def info(self, message):
        self.logger.info(message)


    def warn(self, message):
        self.logger.warning(message)


    def error(self, message):
        self.logger.error(message)


    def load_model(self, model):
        """Load a model from a file if necessary.
        """
        res = model
        if isinstance(model, str):
            if os.path.isfile(model):
                self.info("Loading model from file '{}'".format(model))
                res = load_model(model)
            else:
                self.error("Loading model from file '{}': file not found".format(model))
        return res


    def model_is_valid(self, model=None):
        model = model or self.model
        if not model:
            return False

        return True        
         
    # test with https://github.com/osmr/imgclsmob/blob/master/README.md
    def analyze(self, model=None, layers=[], min_size=50, max_size=0,
                compute_alphas=False, compute_lognorms=True,
                plot=False):
        """
        Analyze the weight matrices of a model.

        layers:
            List of layer ids. If empty, analyze all layers (default)
        min_size:
            Minimum weight matrix size to analyze
        max_size:
            Maximum weight matrix size to analyze (0 = no limit)
        compute_alphas:
            Compute the power laws (alpha) of the weight matrices. 
            Time consuming so disabled by default (use lognorm if you want speed)
        compute_lognorms:
            Compute the log norms of the weight matrices.
        """
        model = model or self.model
        
        res = {}

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
            self.info("Layer {}: {}".format(i+1, l))
            res[i] = {"id": i}
            res[i]["type"] = l
            
            weights = []

            # Filter out layers by numerical id (if any provided)
            if (len(layer_ids) > 0 and (i not in layer_ids)):
                msg = "Skipping (Layer id not requested to analyze)"
                self.info("Layer {}: {}".format(i+1, msg))
                res[i]["message"] = msg
                continue

            # DENSE layer (Keras) / LINEAR (pytorch)
            if isinstance(l, keras.layers.core.Dense) or isinstance(l, nn.Linear):

                res[i]["layer_type"] = LAYER_TYPE.DENSE

                # Filter out layers by type (if any provided)
                if (len(layer_types) > 0 and
                        not any(layer_type & LAYER_TYPE.DENSE for layer_type in layer_types)):
                    msg = "Skipping (Layer type not requested to analyze)"
                    self.info("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue
                
                if isinstance(l, nn.Linear):
                    # pyTorch
                    weights = [np.array(l.weight.data.clone().cpu())]
                else:
                    # keras
                    weights = l.get_weights()[0:1] # keep only the weights and not the bias
                    # TODO: add option to append bias matrix
                    #if add_bias:
                    #    weights = weigths[0]+weights[1]

            # CONV1D layer
            elif (isinstance(l, keras.layers.convolutional.Conv1D)):

                res[i] = {"layer_type": LAYER_TYPE.CONV1D}

                if (len(layer_types) > 0 and
                        not any(layer_type & LAYER_TYPE.CONV1D for layer_type in layer_types)):
                    msg = "Skipping (Layer type not requested to analyze)"
                    self.info("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue
                
                weights = l.get_weights()[0:1] # keep only the weights and not the bias

            # CONV2D layer
            elif isinstance(l, keras.layers.convolutional.Conv2D) or isinstance(l, nn.Conv2d):

                res[i] = {"layer_type": LAYER_TYPE.CONV2D}

                if (len(layer_types) > 0 and
                        not any(layer_type & LAYER_TYPE.CONV2D for layer_type in layer_types)):
                    msg = "Skipping (Layer type not requested to analyze)"
                    self.info("Layer {}: {}".format(i+1, msg))
                    res[i]["message"] = msg
                    continue
                
                if isinstance(l, nn.Conv2d):
                    w = [np.array(l.weight.data.clone().cpu())]
                else:
                    w = l.get_weights()
                    
                weights = self.get_conv2D_Wmats(w[0])

            else:
                msg = "Skipping (Layer not supported)"
                self.info("Layer {}: {}".format(i+1, msg))
                res[i]["message"] = msg
                continue

            self.info("Layer {}: Analyzing {} weight matrices...".format(i+1, len(weights)))

            results = self.analyze_weights(weights, min_size=min_size, max_size=max_size,
                                           compute_alphas=compute_alphas, compute_lognorms=compute_lognorms,
                                           plot=plot)
            if not results:
                msg = "No weigths to analyze"
                self.info("Layer {}: {}".format(i+1, msg))
                res["message"] = msg
            else:
                res[i] = {**res[i], **results}
                
        self.results = res
        
        self.print_results(results=res)

        return res
    
    
    def print_results(self, results=None):
        import numpy as np
        
        if results is None:
            results = self.results

        if not results:
            self.warn("No results to print")
            return

        self.info("### Printing results ###")
            
        lognorms = [] # All lognorms of all slices
        lognorms_compound = [] # Lognorms of the average of the slices
        for layer_id, result in results.items():
            compound = 0
            count = 0
            for slice_id, summary in result.items():
                if not str(slice_id).isdigit() or "lognorm" not in summary:
                    continue
                lognorm = summary["lognorm"]
                lognorms.append(lognorm)
                compound += lognorm
                count += 1

            if count > 0:
                compound = compound / count
                lognorms_compound.append(compound)

                if count > 1:
                    # Compound norm of the multiple slices (conv2D)
                    self.debug("Layer {}: Lognorm compound: {}".format(layer_id, compound))
                else:
                    # No slices (Dense or Conv1D)
                    self.debug("Layer {}: Lognorm: {}".format(layer_id, compound))

        if len(lognorms) > 0:
            avg = np.mean(lognorms)
            self.summary["lognorm"] = avg
            self.info("LogNorm: min: {}, max: {}, avg: {}".format(min(lognorms), max(lognorms), avg))

            avg = np.mean(lognorms_compound)
            self.summary["lognorm_compound"] = avg
            self.info("LogNorm compound: min: {}, max: {}, avg: {}".format(min(lognorms_compound), max(lognorms_compound), avg))
        
        alphas = [] # All alphas for all slices
        alphas_compound = [] # Alphas of the average of the slices
        for layer_id, result in results.items():
            compound = 0
            count = 0
            for slice_id, summary in result.items():
                if not str(slice_id).isdigit() or "alpha" not in summary:
                    continue
                alpha = summary["alpha"]
                alphas.append(alpha)
                compound += alpha
                count += 1
                
            if count > 0:
                compound = compound / count
                alphas_compound.append(compound)

                if count > 1:
                    # Compound alpha of the multiple slices (conv2D)
                    self.debug("Layer {}: Alpha compound: {}".format(layer_id, compound))
                else:
                    # No slices (Dense or Conv1D)
                    self.debug("Layer {}: Alpha: {}".format(layer_id, compound))

        if len(alphas)>0:
            avg = np.mean(alphas)
            self.summary["alpha"] = avg
            self.info("Alpha: min: {}, max: {}, avg: {}".format(min(alphas), max(alphas), avg))

            avg = np.mean(alphas_compound)
            self.summary["alpha_compound"] = avg
            self.info("Alpha compound: min: {}, max: {}, avg: {}".format(min(alphas_compound), max(alphas_compound), avg))

        alphas_weighted = [] # All alphas for all slices
        alphas_weighted_compound = [] # Alphas of the average of the slices
        for layer_id, result in results.items():
            compound = 0
            count = 0
            for slice_id, summary in result.items():
                if not str(slice_id).isdigit() or "alpha_weighted" not in summary:
                    continue
                alpha = summary["alpha_weighted"]
                alphas_weighted.append(alpha)
                compound += alpha
                count += 1
                
            if count > 0:
                compound = compound / count
                alphas_weighted_compound.append(compound)

                if count > 1:
                    # Compound alpha of the multiple slices (conv2D)
                    self.debug("Layer {}: Alpha Weighted compound: {}".format(layer_id, compound))
                else:
                    # No slices (Dense or Conv1D)
                    self.debug("Layer {}: Alpha Weigthed: {}".format(layer_id, compound))

        if len(alphas_weighted)>0:
            avg = np.mean(alphas_weighted)
            self.summary["alpha_weighted"] = avg
            self.info("Alpha Weighted: min: {}, max: {}, avg: {}".format(min(alphas_weighted), max(alphas_weighted), avg))

            avg = np.mean(alphas_weighted_compound)
            self.summary["alpha_weighted_compound"] = avg
            self.info("Alpha Weighted compound: min: {}, max: {}, avg: {}".format(min(alphas_weighted_compound), max(alphas_weighted_compound), avg))

    
    def get_summary(self):
        return self.summary


    def get_conv2D_Wmats(self, Wtensor):
        """Extract W slices from a 4 index conv2D tensor of shape: (N,M,i,j) or (M,N,i,j).  
        Return ij (N x M) matrices
        
        """
        Wmats = []
        s = Wtensor.shape
        N, M, imax, jmax = s[0],s[1],s[2],s[3]
        if N + M >= imax + jmax:
            self.info("Pytorch tensor shape detected: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))
            
            for i in range(imax):
                for j in range(jmax):
                    W = Wtensor[:,:,i,j]
                    if N < M:
                        W = W.T
                    Wmats.append(W)
        else:
            N, M, imax, jmax = imax, jmax, N, M          
            self.info("Keras tensor shape detected: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))
            
            for i in range(imax):
                for j in range(jmax):
                    W = Wtensor[i,j,:,:]
                    if N < M:
                        W = W.T
                    Wmats.append(W)
            
        return Wmats

    
    def analyze_weights(self, weights, min_size=50, max_size=0,
                        compute_alphas=False, compute_lognorms=True,
                        plot=False):
        """Analyzes weight matrices.
        
        Example in Keras:
            weights = layer.get_weights()
            analyze_weights(weights)
        """
        from sklearn.decomposition import TruncatedSVD

        res = {}
        count = len(weights)
        if count == 0:
            return res
        
        alpha_min = 1.5
        alpha_max = 3.5

        for i, W in enumerate(weights):
            res[i] = {}
            M, N = np.min(W.shape), np.max(W.shape)
            Q=N/M
            res[i]["N"] = N
            res[i]["M"] = M
            res[i]["Q"] = Q

            if M < min_size:
                summary = "Weight matrix {}/{} ({},{}): Skipping: too small (<{})".format(i+1, count, M, N, min_size)
                res[i]["summary"] = summary 
                self.info("    {}".format(summary))
                continue

            if max_size > 0 and M > max_size:
                summary = "Weight matrix {}/{} ({},{}): Skipping: too big (testing) (>{})".format(i+1, count, M, N, max_size)
                res[i]["summary"] = summary 
                self.info("    {}".format(summary))
                continue
                
            self.info("    Weight matrix {}/{} ({},{}): Analyzing ..."
                     .format(i+1, count, M, N))
            
            if compute_alphas:

                svd = TruncatedSVD(n_components=M-1, n_iter=7, random_state=10)
                svd.fit(W) 
                sv = svd.singular_values_
                evals = sv*sv

                # Other (slower) way of computing the eigen values:
                #X = np.dot(W.T,W)/N
                #evals2 = np.linalg.eigvals(X)
                #res[i]["lambda_max2"] = np.max(evals2)

                lambda_max = np.max(evals)
                fit = powerlaw.Fit(evals, xmax=lambda_max, verbose=False)
                alpha = fit.alpha
                res[i]["alpha"] = alpha
                D = fit.D
                res[i]["D"] = D
                res[i]["lambda_min"] = np.min(evals)
                res[i]["lambda_max"] = lambda_max
                alpha_weighted = alpha * np.log10(lambda_max)
                res[i]["alpha_weighted"] = alpha_weighted
                tolerance = lambda_max * M * np.finfo(np.max(sv)).eps
                res[i]["rank_loss"] = np.count_nonzero(sv > tolerance, axis=-1)

                summary = "Weight matrix {}/{} ({},{}): Alpha: {}, Alpha Weighted: {}, D: {}".format(i+1, count, M, N, alpha, alpha_weighted, D)
                res[i]["summary"] = summary 
                self.info("    {}".format(summary))

                if alpha < alpha_min or alpha > alpha_max:
                    message = "Weight matrix {}/{} ({},{}): Alpha {} is in the danger zone ({},{})".format(i+1, count, M, N, alpha, alpha_min, alpha_max)
                    self.info("    {}".format(message))

                if plot:
                    fig2 = fit.plot_pdf(color='b', linewidth=2)
                    fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig2)
                    fit.plot_ccdf(color='r', linewidth=2, ax=fig2)
                    fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig2)
                    plt.title("Power law fit for Weight matrix {}/{}".format(i+1, count))
                    plt.show()

                    # plot eigenvalue histogram
                    plt.hist(evals, bins=100, density=True)
                    plt.title(r"ESD (Empirical Spectral Density) $\rho(\lambda)$" + " for Weight matrix {}/{}".format(i+1, count))
                    plt.show()

                    plt.loglog(evals)
                    plt.title("Eigen Values for Weight matrix {}/{}".format(i+1, count))
                    plt.show()
                
            if compute_lognorms:
                lognorm = np.log10(np.linalg.norm(W))
                res[i]["lognorm"] = lognorm
                
                summary = "Weight matrix {}/{} ({},{}): Lognorm: {}".format(i+1, count, M, N, lognorm)
                res[i]["summary"] = summary 
                self.info("    {}".format(summary))

        return res
