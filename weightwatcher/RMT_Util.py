# -*- coding: utf-8 -*-
import logging
from copy import deepcopy
import pickle, time
from shutil import copy
import sys, os
import warnings

import powerlaw
from scipy import optimize, stats
from sklearn.neighbors import KernelDensity
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as stats

from .constants import *
from .WW_powerlaw import *


logger = logging.getLogger(WW_NAME)


def has_mac_accelerate():
    """Check if the platform is the MAC M1/M2 and numpy is using the accelerate library.
    If so, numpy SVD will be faster than linalg SVD, and we auto-switch to that upon import
    
    Note: """
    
    import platform
    import warnings
    
    use_linalg = False
    mac_arch = platform.machine()
    if mac_arch == 'arm64':
    
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            import numpy.distutils.system_info as sysinfo
            info = sysinfo.get_info('accelerate')
            if info is not None and len(info)>0:
                for x in info['extra_link_args']:
                    if 'Accelerate' in x:
                        use_linalg = True
                        
    return use_linalg
    


# PyTorch has 2 separate SVD methods, one for the vals, and one for the factorization
# https://pytorch.org/docs/stable/generated/torch.linalg.eig.html
# https://pytorch.org/docs/stable/generated/torch.linalg.svd.html
# https://pytorch.org/docs/stable/generated/torch.linalg.svdvals.html

# TODO: change this based on ENV var and/or auto-detect
# see: issue #220
if has_mac_accelerate():
    logger.info("Using Numoy with MAC M1/2 Accelerate for SVD")
    _eig_full_accurate = lambda W: np.linalg.eig(W)
    _svd_full_accurate = lambda W: np.linalg.svd(W, compute_uv=True)
    _svd_vals_accurate = lambda W: np.linalg.svd(W, compute_uv=False)
else:
    logger.info("Using Scipy for SVD")
    _eig_full_accurate = lambda W: sp.linalg.eig(W)
    _svd_full_accurate = lambda W: sp.linalg.svd(W, compute_uv=True)
    _svd_vals_accurate = lambda W: sp.linalg.svd(W, compute_uv=False)
    
    
    
def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    return False


"""This method is over-written in tourch cuda is available"""
torch_T_to_np = lambda T: T.to("cpu").float().numpy()

# 
try:
    import torch
    if torch.cuda.is_available() :
        logger.info("Torch CUDA available, using torch_wrapper")

        # float changed to half to save memory
        torch_T_to_np = lambda T: T.to("cpu").half().numpy()
         
        def torch_wrapper(M, f):
            torch.cuda.empty_cache()
            with torch.no_grad():
                M_cuda = torch.Tensor(M).to("cuda")
                rvals = f(M_cuda)
            del M_cuda
            return rvals          
            

        def _eig_full_fast(M):
            L, V = torch_wrapper(M, lambda M: torch.linalg.eig(M))
            return torch_T_to_np(L), torch_T_to_np(V)
        def _svd_full_fast(M):
            U, S, Vh = torch_wrapper(M, lambda M: torch.linalg.svd(M))
            return torch_T_to_np(U), torch_T_to_np(S), torch_T_to_np(Vh)
        def _svd_vals_fast(M):
            S = torch_wrapper(M, lambda M: torch.linalg.svdvals(M))
            return torch_T_to_np(S)
    else:
        msg_svd = "SciPy"
        if has_mac_accelerate():
            msg_svd = "NumPy"

        logger.warning(f"PyTorch is available but CUDA is not. Defaulting to {msg_svd} for SVD")
        raise ImportError()
    
except ImportError:
    # if torch / cuda are not available, default to scipy
    _eig_full_fast = _eig_full_accurate
    _svd_full_fast = _svd_full_accurate
    _svd_vals_fast = _svd_vals_accurate
    
    torch_T_to_np = lambda T: T.to("cpu").float().numpy()

    

def eig_full(W, method=ACCURATE_SVD):
    assert method.lower() in [ACCURATE_SVD, FAST_SVD], method #TODO TRUNCATED_SVD?
    if method == ACCURATE_SVD: return _eig_full_accurate(W)
    if method == FAST_SVD:     return _eig_full_fast(W)

def svd_full(W, method=ACCURATE_SVD):
    assert method.lower() in [ACCURATE_SVD, FAST_SVD], method #TODO TRUNCATED_SVD
    if method == ACCURATE_SVD: return _svd_full_accurate(W)
    if method == FAST_SVD:     return _svd_full_fast(W)

def svd_vals(W, method=ACCURATE_SVD):
    assert method.lower() in [ACCURATE_SVD, FAST_SVD], method #TODO TRUNCATED_SVD
    if method == ACCURATE_SVD: return _svd_vals_accurate(W)
    if method == FAST_SVD:     return _svd_vals_fast(W)




# ## Generalized Entropy
# Trace Normalization
#def matrix_entropy_(W):
#   """Matrix entropy of W real rectangular matrix, computed using the singular values; may be slow"""
#    W = W / np.trace(W)
#    N, M  = np.min(W.shape), np.max(W.shape)
#    u, sv, vh = svd(W)  
#    
#    evals  = sv * sv
#    return matrix_entropy(evals, N)
def matrix_rank(svals, N, tol=None):
    """Matrix rank, computed from the singular values directly

    svals are the singular values
    N is the largest dimension of the matrix
 
    re-implements np.linalg.matrix_rank(W) """
    S = svals
    if tol is None:
        tol = np.max(S) * N * np.finfo(S.dtype).eps
    return np.count_nonzero(S > tol)


def calc_rank_loss(svals, N, tol=None):
    """Rank loss for this matrix, from the singular values and the largest dim N."""

    rank = matrix_rank(svals, N, tol=tol) #np.linalg.matrix_rank(W)
    return len(svals) - rank

def matrix_entropy(svals, N):
    """Matrix entropy of real, computed using the singular values, and the dim N"""


    rank = matrix_rank(svals, N) #np.linalg.matrix_rank(W)

    evals = svals*svals
    p = evals / np.sum(evals)
    if (rank == 1):
        rank = 1.000001
    entropy = -np.sum(p * np.log(p)) / np.log(rank) 
    return entropy

# Wigner SemiCircle Plots


def marchenko_pastur_pdf(x_min, x_max, Q, sigma=1.0, grid_size=0.001):
    """Return a discrete set of (x,y) values representing the Marchenko-Pastur distrbution from RMT.
    Computing on a grid of size 0.001 by default"""
    y = 1 / Q

    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)  # Largest eigenvalue
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)  # Smallest eigenvalue

    x_min = max(x_min, a)
    x_max = min(x_max, b)
    x = np.arange(x_min, x_max, grid_size)

    return x, (1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt((b - x) * (x - a))


def quarter_circle_pdf(x_min, x_max, sigma=1.0, grid_size=0.001):
    """Return a discrete set of (x,y) values representing the Quarter-Circle distribution from RMT.
    Computing on a grid of size 0.001 by default"""
    x = np.arange(x_min, min(x_max, 2), grid_size)

    # When Q = 1, we look at the singular values instaed of eigenvalues
    return x, (1 / (np.pi * sigma * sigma)) * np.sqrt((4 - x ** 2)+10**-9)


def calc_sigma(Q, evals):
    """Given input aspect ratio Q, and eigenvalues (evals), compute the sigma from the Marchenko-Pastur RMT."""
    lmax = np.max(evals)
    inv_sqQ = 1.0 / np.sqrt(Q)
    sigma_2 = lmax / np.square(1 + inv_sqQ)
    sigma = np.sqrt(sigma_2)
    return sigma


def calc_lambda_plus(Q, sigma):
    """Given input aspect ratio Q, and variance (sigma), compute the bulk edge maximum eigenvalue from the Marchenko-Pastur RMT."""
    return np.power(sigma * (1 + np.sqrt(1 / Q)), 2)


def calc_lambda_minus(Q, sigma):
    """Given input aspect ratio Q, and variance (sigma), compute the bulk edge minimum eigenvalue from the Marchenko-Pastur RMT."""
    return np.power(sigma * (1 - np.sqrt(1 / Q)), 2)


def get_Q(W):
    """Return the apsect ratio Q=N/M of a rectangular matrix W"""
    if W.shape[1] > W.shape[0]:
        M, N = W.shape
    else:
        N, M = W.shape
    Q = N / M
    return Q


def hard_rank(W, sv):
    """hard rank, using tolerance from numerical recipes, not default scipy tolerance"""
    tol = np.max(sv) * np.finfo(np.float32).eps / (2.0 * np.sqrt(np.sum(W.shape) + 1))
    return np.linalg.matrix_rank(W, tol=tol)


# uss FAST SVD method: notice we miss 1 eigenvalue here...using 
def get_shuffled_eigenvalues(W, layer=7, num=100, method=ACCURATE_SVD):
    "get eigenvalues for this model, but shuffled, num times"
    
    N, M = W.shape[0], W.shape[1]       

    if (N < M):
        N, M = W.shape[1], W.shape[0] 
   
    eigenvalues = []
    for idx in tqdm(range(num)):
        W_shuf = W.flatten()
        np.random.shuffle(W_shuf)
        W_shuf = W_shuf.reshape([N, M])

        sv = svd_vals(W_shuf, method=method)

        eigenvalues.extend(sv * sv)
        
    evals = (np.array(eigenvalues).flatten())
    return evals


def plot_density_and_fit(eigenvalues=None, model=None, layer_name="", layer_id=0,
                     Q=1.0, num_spikes=0, sigma=None,
                     alpha=0.25, color='blue', skip=False, verbose=True, plot=True, cutoff=0.0):
    """Plot histogram of eigenvalues, for Q, and fit Marchenk Pastur.  
    If no sigma, calculates from maximum eigenvalue (minus spikes)
    
    If Q = 1, analyze the singular values for the Quarter Circle law"""
    
    # if eigenvalues is None:
    #    eigenvalues = get_eigenvalues(model, weightfile, layer)
   
    if Q == 1:
        to_fit = np.sqrt(eigenvalues)
        cutoff = np.sqrt(cutoff)
        label = r"$\rho_{emp}(\nu)$"
        title = " W{} SSD, QC Sigma={:0.3}" 
    else:
        to_fit = eigenvalues
        label = r'$\rho_{emp}(\lambda)$'
        title = " W{} ESD, MP Sigma={:0.3}f" 
        
    if plot:
        plt.hist(to_fit, bins=100, alpha=alpha, color=color, density=True, label=label);
        plt.legend()
        
        if cutoff > 0.0:
            plt.axvline(x=cutoff, linewidth=1, color='r', ls='dashed')
    
    if skip:
        return
    if not (num_spikes):
        num_spikes = 0
        
    # sort, descending order, minus a few max to_fit

    if (sigma is None):
        evals = np.sort(eigenvalues)[::-1][num_spikes:]
        sigma = calc_sigma(Q, evals)
        
    percent_mass = 100.0 * (num_spikes) / len(to_fit)

    bulk_fit = np.sort(to_fit)[num_spikes:]
    x_min, x_max = np.min(bulk_fit), np.max(bulk_fit)
    
    if Q == 1:
        x, mp = quarter_circle_pdf(x_min, x_max, sigma)
    else:
        x, mp = marchenko_pastur_pdf(x_min, x_max, Q, sigma)

    if plot:
        plt.title(title.format(layer_name, sigma))
        plt.plot(x, mp, linewidth=1, color='r', label="MP fit")
        
    if verbose:
        print("% spikes outside bulk {0:.2f}".format(percent_mass))
        print("% sigma {0:.4f}".format(sigma))
        
    # TODO 
    # so fit can be plot on different scale (log,) etc
    # expose for debugging
    return sigma, x, mp
    
    #return sigma


def plot_density(to_plot, sigma, Q, method="MP", color='blue', cutoff=0.0):
    """Method = 'MP' or 'QC'
    
    """

    if method == "MP":
        to_plot = np.sort(to_plot)
        x_min, x_max = np.min(to_plot), np.max(to_plot)
        x, y = marchenko_pastur_pdf(x_min, x_max, Q, sigma)
    elif method == "QC":
        cutoff = np.sqrt(cutoff)
        to_plot = np.sort(to_plot)
        x_min, x_max = np.min(to_plot), np.max(to_plot)
        x, y = quarter_circle_pdf(x_min, x_max, sigma)
        
    plt.hist(to_plot, bins=100, alpha=0.6, color=color, density=True, label="ead")
    plt.plot(x, y, linewidth=1, color='r', label = method + " fit")
    plt.legend()
    
    if cutoff > 0.0:
        plt.axvline(x=cutoff, linewidth=1, color='r', ls='dashed')
    
    return None

# # Scree Plots


# Eigenvalues for correlation weight matrices, for last 2 layers
# for Scree plots
# check normalization...maybe replace with svd approach above
def matrix_eigenvalues(model, layer=2):    
    W = model.layers[layer].get_weights()[0]
    W = W / np.linalg.norm(W)
    WW = np.dot(W.transpose(), W)  # /float(N7)
    evs, _ = np.linalg.eig(WW)    
    return evs


# TODO: refactor
def scree_plot(model, weightfile, layer=2, color='blue', label=''):    
    model.load_weights(weightfile)
    evs = matrix_eigenvalues(model, layer)
    eigvals = np.flip(np.sort(evs), axis=0)
    sing_vals = np.arange(len(eigvals)) + 1
    plt.plot(sing_vals, eigvals, color, linewidth=1, label=label)

# # Soft / Stable Rank


def stable_rank(evals):
    """compute the matrix soft rank (or stable rank), given eigenvalues (evals) for a matrix W
    See:  https://arxiv.org/abs/1810.01075 """ 
    return np.sum(evals) / np.max(evals)


def matrix_soft_rank(W, method=ACCURATE_SVD):
    """compute the matrix soft rank (or stable rank), given rectangular numpy matrix W""
    See:  https://arxiv.org/abs/1810.01075 """ 
    W = W / np.trace(W)
    sv = svd_vals(W, method=method)
    return stable_rank(sv * sv)


def mp_soft_rank(evals, num_spikes):
    """compute the MP soft rank, given a set of eigenvalues, and the number of spikes outside the MP bulk region
    See:  https://arxiv.org/abs/1810.01075 """ 
    
    evals = np.array(evals)
    lambda_max = np.max(evals)
    if num_spikes > 0:
        evals = np.sort(evals)[::-1][num_spikes:]
        lambda_plus = np.max(evals)
    else:
        lambda_plus = lambda_max
        
    return lambda_plus / lambda_max


def calc_mp_soft_rank(evals, Q, sigma):
    """compute the MP soft rank, given a set of eigenvalues, aspect ratio, and MP variance (sigma)
    See:  https://arxiv.org/abs/1810.01075 """ 
    lambda_plus = calc_lambda_plus(Q, sigma)
    lambda_max = np.max(evals)
  
    return lambda_plus / lambda_max

# ## Eigenvector Localization


def localization_ratio(vec):
    """compute the vector localization_ratio this numpy vector.
    See:  https://arxiv.org/abs/1810.01075"""
    return np.linalg.norm(vec, ord=1) / np.linalg.norm(vec, ord=np.inf)


def participation_ratio(vec):
    """compute the vector localization_ratio this numpy vector.
    See:  https://arxiv.org/abs/1810.01075"""
    return np.linalg.norm(vec, ord=2) / np.linalg.norm(vec, ord=4)


def vector_entropy(u):
    """Vector entropy, as in the  Porter-Thomas distribution
    
    see:  https://arxiv.org/pdf/cond-mat/9810255.pdf
    """
    u2 = np.linalg.norm(u) / 2
    entropy = np.sum(np.exp(-u2) / np.sqrt(2.0 * np.pi))
    return entropy


def discrete_entropy(vec, num_bins=100):
    """compute the discrete vector entropy  for this numpy vector, given the number of bins"""

    vec = np.absolute(vec)
    vec = vec - np.mean(vec)
    h = np.histogram(vec, density=True, bins=num_bins)[0];
    p = np.array(h) + 0.0000000001
    
    p = p / np.sqrt(2 * np.pi)
    p = p / np.sum(p)

    # p = p/(2*np.pi)
    entropy = -np.sum(p * np.log(p))
    entropy = entropy  # /(2*np.pi)#/float(num_bins)
    return entropy


def max_discrete_entropy(len_vec, num_bins=100, sample_size=100000):
    """compute maximum possible entropy for this numpy vector length and bin sizes"""
    entropies = []
    for i in tqdm(range(sample_size)):
        test_vec = np.random.normal(0, 1, len_vec)
        s = discrete_entropy(test_vec, num_bins=num_bins)
        entropies.append(s)

    return np.max(entropies)

# ## Misc


def line_entropy(data):
    """Compute the entropy of a 1d function
    
    see issue #173 """
    
    # Discretize the data
    hist, bin_edges = np.histogram(data, bins='auto')
    # Compute the probability density of the function
    prob = hist / np.sum(hist)
    # Compute the entropy of the function
    the_entropy = stats.entropy(prob)
    return the_entropy



def spike_min(Q):
    """minimum perturbation to W to see a spike"""
    return 1 / np.sqrt(np.sqrt(Q))


def spike_lmax(S, Q):
    """Maximum spike given a perturbation"""
    S2 = S * S
    return ((1.0 / Q) + S2) * (1 + (1.0 / S2))

# ## Fit PowerLaw


def mu4alpha(alpha):
    if (alpha >= 2.0 and alpha <= 3.0):
        mu = 2.0
    elif (alpha < 2.0):
        mu = 2.0 * (alpha - 1.0)
    elif (alpha > 3.0):
        mu = alpha - 1.0
    return mu


def best_dist(fit):
    """From a powerlaw fit, find the best distribution (see powerlaw package) """
    distName = 'power_law'
    dist = "PL"

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
        R, p = fit.distribution_compare('truncated_power_law', 'power_law', normalized_ratio=True)
        if R > 0 and p <= 0.05:
            distName = 'truncated_power_law'
            dist = 'TPL'
            
        R, p = fit.distribution_compare(distName, 'exponential', normalized_ratio=True)
        if R < 0 and p <= 0.05:
            dist = 'EXP'
            return dist
    
        R, p = fit.distribution_compare(distName, 'stretched_exponential', normalized_ratio=True)
        if R < 0 and p <= 0.05:
            dist = 'S_EXP'
            return dist
            
        R, p = fit.distribution_compare(distName, 'lognormal', normalized_ratio=True)
        if R < 0 and p <= 0.05:
            dist = 'LOG_N'
            return dist

    return dist

# def fit_powerlaw(evals, verbose=True):
#    fit = powerlaw.Fit(evals, xmax=np.max(evals))
#    return [fit.alpha, fit.D, best_dist(fit)]A

# ## Auto fit MP

#### Using Kernel Density Estimator




def marchenko_pastur_fun(x, Q, sigma=1):
    y = 1 / Q
    
    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)  # Largest eigenvalue
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)  # Smallest eigenvalue

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        val = np.nan_to_num((1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt((b - x) * (x - a)))

    return x, val, a, b


def quarter_circle_fun(x, sigma=1): 

    # When Q = 1, we look at the singular values instead of eigenvalues
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        val = np.nan_to_num((1 / (np.pi * sigma * sigma)) * np.sqrt(4 - x ** 2))

    return x, val


def resid_mp(p, evals, Q, bw, allresid=True, num_spikes=0, debug=False):  
    "residual that floats sigma but NOT Q or num_spikes YET, 10% cutoff each edge"
    sigma = p

#    if (sigma > 1): #sigma must be less than 1
#        resid = np.zeros(1000) + 1000
#        if allresid:
#            return resid
#        else:
#            return np.sum(resid**2)

    # kernel density estimator
    kde = KernelDensity(kernel='linear', bandwidth=bw).fit(evals.reshape(-1, 1))
    xde = np.linspace(0, np.max(evals) + 0.5, 1000)
    X_plot = xde[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)
    yde = np.exp(log_dens)   

    # hack to try to fix fits
    # remove areas where the density is almost zero
    THRESH = 0.01
    bad_ids = np.where(np.array(yde) < THRESH)[0]
    good_ids = np.where(np.array(yde) > THRESH)[0]
    xde = np.array(xde)[good_ids]
    yde = np.array(yde)[good_ids]

    if Q == 1:
        # Quarter Cirle Law fit for this sigma
        xmp, ymp = quarter_circle_fun(xde, sigma=sigma)
        resid = ymp - yde
    else:
        # MP fit for this sigma
        xmp, ymp, a, b = marchenko_pastur_fun(xde, Q=Q, sigma=sigma)

        ### issue #60  change if statement
        #if (b > max(xde)) | (a > xde[np.where(yde == max(yde))][0]) | (sigma > 1):
        if (b < xde[np.where(yde == max(yde))][0] ) | (b > max(xde)) | (a > xde[np.where(yde == max(yde))][0]):
            resid = np.zeros(len(yde)) + 1000
        else:
            # form residual, remove nan's 
            resid = ymp - yde
    #     resid = np.nan_to_num(resid)

    if debug:
        plt.plot(xde, yde, color='cyan')
        plt.plot(xmp, ymp, color='orange')
        plt.axhline(y=THRESH)
        plt.show(); plt.clf()
        print("sigma {}  mean residual {}".format(sigma, np.mean(resid)))

    # hack to try to fix fits
    bad_zeros = np.zeros(len(bad_ids))
    resid = np.concatenate((resid,bad_zeros))
    
    if allresid:
        return resid
    else:
        return np.sum(resid ** 2)


def shuf_matrix(W, seed=None):
    """Make a copy of the input matrix W and shuffle it's elements"""
    w = W.copy()

    n, m = w.shape    
    w_flat = w.flatten()

    if seed is not None:
        np.random.seed(seed)  # for reproduction

    w_shuf = np.random.choice(w_flat, size=int(n * m)).reshape((n, m))
    
    return w_shuf


def fit_density(evals, Q, bw=0.1, sigma0=None):
    "Fit the esd to a MP distribution: simple fit of evals, only floats sigma right now"
    
    if sigma0 is None:
        sigma0 = 1.0
        
    if Q == 1:
        to_fit = np.sqrt(evals)
    else:
        to_fit = evals
        
    [sigma1], cov, infodict, mesg, ierr = optimize.leastsq(resid_mp, [sigma0],
                                args=(to_fit, Q, bw), full_output=True)
    return sigma1, infodict['fvec']


def fit_density_with_range(evals, Q, bw=0.1, sigma_range=(slice(0.1, 1.25, 0.01),) ):
    assert type(sigma_range) == tuple, ValueError("sigma_range must be tuple")
    assert type(sigma_range[0]) == slice

    ### issue #60 fix
    #   reset sigma range
    sigma_scaling_factor = calc_sigma(Q, evals)
    sigma_range = (slice(0.05 * sigma_scaling_factor, 2.0 * sigma_scaling_factor, 0.01 * sigma_scaling_factor),)
    #
    ###

    if Q == 1:
        to_fit = np.sqrt(evals)
    else:
        to_fit = evals
        
    brute_output = optimize.brute(resid_mp, sigma_range,
                                  args=(to_fit, Q, bw, False), full_output=True)
    
    return brute_output[0][0], brute_output[1]  # sigma_optimized, resid

# import pandas as pd
# def fit_mp_findspikes(evals, Q):
#     '''Remove eigen (spikes) from largest to smallest'''
#     evals = sorted(evals)[::-1]
#     
#     df_output = np.zeros((len(evals), 3))
#     for spike in range(len(evals)):
#         this_evals = np.array(evals[spike:])
#         sigma, fvec = fit_mp(this_evals, Q)
#         df_output[spike, 0] = spike
#         df_output[spike, 1] = sigma
#         df_output[spike, 2] = np.linalg.norm(fvec)
#         
#     return pd.DataFrame(df_output, columns = ['spikes', 'sigma', 'F_norm'])


def plot_loghist(x, bins=100, xmin=None):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins, density=True)

    if xmin:
        plt.axvline(xmin, color='r', label=r'$\lambda_{min}$')

    plt.xscale('log')
    
    
def permute_matrix(W):
    """permute a matrix in a reversible way"""
    
    num_params = np.prod(W.shape)
    vec = W.reshape(num_params)
    p_ids = np.random.permutation(np.arange(num_params))
    p_vec = vec[p_ids]
    p_W = p_vec.reshape(W.shape)
            
    return p_W, p_ids


def unpermute_matrix(W, p_ids):
    """unpermute a matrix, using the original ids to permute it"""
    
    num_params = np.prod(W.shape)
    vec = W.reshape(num_params)
    unp_ids = np.argsort(p_ids)
    unp_vec = vec[unp_ids]
    unp_W = unp_vec.reshape(W.shape)
    
    return unp_W


def save_fig(plt, figname, layer_id, savedir):
    """Save the figure to the savedir directory. 
       If directory is not present, create it
       """
    
    figname = "{}/ww.layer{}.{}.png".format(savedir, layer_id, figname)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    plt.tight_layout()
    plt.savefig(figname)
    return 
 
#TODO: WHAT IS THIS?
def fit_xxx_powerlaw(evals, xmin=None):
    N = int(len(evals)/4)
    xmax = np.max(evals)       
   
    if xmin is not None and xmin != -1:
        prev_fit = powerlaw.Fit(evals, xmin=xmin, xmax=xmax)
    else:
        prev_fit = powerlaw.Fit(evals, xmax=xmax)
       
    fit = prev_fit 

    prev_alpha = prev_fit.alpha
    prev_sigma = prev_fit.sigma

    first_fit = prev_fit
    
               
    return fit
    
# check alpha is decreasing
# DO WE NEED XMAX='force'?
def fit_clipped_powerlaw(evals, xmin=None, xmax=None, verbose=False, max_fingers=DEFAULT_MAX_FINGERS, min_alpha=2.0, 
                         finger_thresh=DEFAULT_FINGER_THRESH, logger=None, plot=False, pl_package=WW_POWERLAW_PACKAGE):
    """Fits a powerlaw only, not a truncated power law
       clips off the max evals until a powerlaw is found, or stops half-way into the ESD
       
       Used to remove power-law tail fingers, which may result from finite size effects
       
       Assumes eval are in sort order
       
       Parameters:
       
         max_fingers:  max number of eigenvalues to clip
       
         min_alpha:  stops if alpha drops below min_alpha
         
         finger_thresh=1.0   alpha has to drop this much to stop
        
        NOTE: currently this only works for the fit='power law'
       
        """
        
    if logger is None:
        import logging
        logger = logging.getLogger("RMT_Util")
        logger.setLevel(logging.INFO)
                
    logger.info(f"fit_clipped_powerlaw: max_fingers={MAX_FINGERS} xmax={xmax}")
    fit = None
    #with warnings.catch_warnings():
        #warnings.simplefilter(action='ignore', category=RuntimeWarning) 
            
    #assert(evals[-1]> evals[0]) 
    # xmaxxmax = None #np.max(evals)

    if xmin is not None and xmin != -1:
        #prev_fit =  powerlaw.Fit(evals, xmin=xmin, xmax=xmax, verbose=verbose)
        prev_fit =  pl_fit(data=evals, xmin=xmin, xmax=xmax, verbose=verbose, distribution=POWER_LAW, pl_package=pl_package)
    else:
        #prev_fit = powerlaw.Fit(evals, xmax=xmax)
        prev_fit =  pl_fit(data=evals, xmax=xmax, verbose=verbose, distribution=POWER_LAW, pl_package=pl_package)

    
    # NOT USED ANYMORE
    #R = 1.0
    #p = 0.01
    #R, p = prev_fit.distribution_compare('truncated_power_law', 'power_law', normalized_ratio=True)
    #logger.info(f"fit alpha {prev_fit.alpha:0.2f} sigma {prev_fit.sigma:0.2f} TPL or PL? {R:0.4f}")     

    prev_alpha = prev_fit.alpha
    prev_sigma = prev_fit.sigma
    #prev_R = R
    first_fit = prev_fit
    best_fit = first_fit

    num_fingers = 0
    
    for idx in range(2,max_fingers):
        if xmax is not None and xmax is not False:
            xmax = np.max(evals[-idx])
        
         
        if xmin is not None and xmin != -1:
            #fit = powerlaw.Fit(evals, xmin=xmin, xmax=xmax, verbose=verbose)
            fit = pl_fit(data=evals[:-idx], xmin=xmin, xmax=xmax, verbose=verbose, distribution=POWER_LAW, pl_package=pl_package)
        else:
            #fit = powerlaw.Fit(evals, xmax=xmax, verbose=verbose)
            fit = pl_fit(data=evals[:-idx], xmax=xmax, verbose=verbose, distribution=POWER_LAW, pl_package=pl_package)
  
        
        # this is only meaningful if the orginal fit is a TPL and the new fit is a PL
        # stop when distribution becomes power law
        #R, p = fit.distribution_compare('truncated_power_law', 'power_law', normalized_ratio=True)
        logger.info(f"fix_fingers: idx={idx} fit alpha {fit.alpha:0.4f}  sigma {fit.sigma:0.4f} xmax={xmax}")     
        #if R > 0.0:
        #    break
   
        if ((fit.alpha + finger_thresh) < first_fit.alpha) : #and fit.sigma < prev_sigma:
            logger.info(f"fix_fingers: best fit found stopping at {idx} {fit.alpha:.2f} << {prev_alpha:0.2f} ")  
            best_fit = fit
            num_fingers = idx - 1

            break
        
        if min_alpha is not None and fit.alpha <= min_alpha: 
            logger.info(f"stopping at min alpha = {fit.alpha:0.2f}")  
            break
        
        
        prev_fit = fit
        prev_alpha = fit.alpha
        prev_sigma = fit.sigma
        #prev_R = R

            
    if idx == max_fingers:
        logger.info("Unable to find smaller alpha, stopping")
        fit = first_fit
    else:
    #  provide the next 3 alpha, and issue a warning if they are out of whack
        logger.info(f"fix_fingers: checking final alpha, starting with idx={idx}, xmax={xmax}")
        for idy in range(idx+1, idx+4):
            data = evals[:-idy]
            if xmax is not None and xmax is not False:
                xmax = np.max(data)
                logger.debug(f"fix_fingers: checking final alpha: idy={idx}, xmax={xmax}")

            # TODO: fix , allows xmax settings
            # if xmax==-1 ?
            if xmin is not None and xmin != -1:
                check_fit = pl_fit(data=data, xmin=xmin, xmax=xmax, verbose=verbose, distribution=POWER_LAW, pl_package=pl_package)
            else:
                check_fit = pl_fit(data=data, xmax=xmax, verbose=verbose, distribution=POWER_LAW, pl_package=pl_package)
            logger.info(f"checking fit {idy}  alpha {fit.alpha:0.2f} sigma {fit.sigma:0.2f}  xmax {xmax}")     

                
            if np.abs(check_fit.alpha - fit.alpha) > finger_thresh :
                logger.warning(f"clipped fit may be spurious, new alpha found: {check_fit.alpha:0.2f}")
                break

    
    return best_fit, num_fingers, first_fit
             
        
# https://medium.com/@sourcedexter/how-to-find-the-similarity-between-two-probability-distributions-using-python-a7546e90a08d
def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (stats.entropy(p, m) + stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def rescale_eigenvalues(evals):
    """Rescale eigenvalues by their Frobenius Norm"""
    
    M = len(evals)
    Wnorm = np.sqrt(np.sum(evals))
    Wscale = np.sqrt(M)/Wnorm
    evals = (Wscale*Wscale)*evals
    
    return evals, Wscale  

def un_rescale_eigenvalues(evals, Wscale):
    """UnRescale eigenvalues """
    
    evals = evals/(Wscale*Wscale)
    return evals

def detX_constraint(evals, rescale=True):
    """Identifies the number of eigenvalues necessary to best satisify the det X = 1 constraint
    
    Parameters
    ----------
    evals : float array
        Eigenvalues, assumed to be in sort order, evals[0]<evals[-1]
        
    rescale: bool
        Rescale the eigenvalues by the Frobenius Norm if not already rescaled
    
    Returns:
    --------
    
        num_evals:  int
        number of eigenvalues in the tail
        
        det_x_idx:  int
        index of the first eigenvalue in the tail
    
    """

    assert(evals[0]<evals[-1])
    num_evals = len(evals)
    idx = 0
    
    
    if rescale:
        Wnorm = np.sqrt(np.sum(evals))
        Wscale = np.sqrt(num_evals)/Wnorm 
        evals = (Wscale*Wscale)*evals

    for idx in range(len(evals)-1, 0, -1):
        detX = np.prod(evals[idx:])
        if detX < 1.0:
            num_evals = len(evals) - idx 
            break    
    
    
    return num_evals, idx


def combine_weights_and_biases(W,b):
    """concat the biases onto the weight matrix if not null or zero"""
    
    Wb = W
    if b is not None and np.sum(b)>ALMOST_ZERO:
        if W.shape[1]==b.shape[0]:
            Wb = np.vstack([W,b])
        else:
            Wb = np.vstack([W.T,b]).T
            
    return Wb



