# -*- coding: utf-8 -*-

import sys, os
import pickle, time
from copy import deepcopy
from shutil import copy
import warnings


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import scipy as sp
from scipy.linalg import svd

from scipy import optimize
from sklearn.neighbors import KernelDensity

import powerlaw
import tqdm

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

    rank = matrix_rank(svals, N) #np.linalg.matrix_rank(W)
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
    x = np.arange(x_min, x_max, grid_size)

    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)  # Largest eigenvalue
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)  # Smallest eigenvalue
    
    return x, (1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt((b - x) * (x - a))


def quarter_circle_pdf(x_min, x_max, sigma=1.0, grid_size=0.001):
    """Return a discrete set of (x,y) values representing the Quarter-Circle distribution from RMT.
    Computing on a grid of size 0.001 by default"""
    x = np.arange(x_min, x_max, grid_size)
    
    # When Q = 1, we look at the singular values instaed of eigenvalues
    return x, (1 / (np.pi * sigma * sigma)) * np.sqrt(4 - x ** 2)


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
def get_shuffled_eigenvalues(W, layer=7, num=100):
    "get eigenvalues for this model, but shuffled, num times"
    
    N, M = W.shape[0], W.shape[1]       

    if (N < M):
        N, M = W.shape[1], W.shape[0] 
   
    eigenvalues = []
    for idx in tqdm(range(num)):
        W_shuf = W.flatten()
        np.random.shuffle(W_shuf)
        W_shuf = W_shuf.reshape([N, M])

        u, sv, vh = svd(W_shuf)

        eigenvalues.extend(sv * sv)
        
    evals = (np.array(eigenvalues).flatten())
    return evals


def plot_density_and_fit(eigenvalues=None, model=None, layer_name="", layer_id=0,
                     Q=1.0, num_spikes=0, sigma=None,
                     alpha=0.25, color='blue', skip=False, verbose=True, plot=True, cutoff=0.0):
    """Plot histogram of eigenvalues, for Q, and fit Marchenk Pastur.  
    If no sigma, calculates from maximum eigenvalue (minus spikes)
    Can read keras weights from model if specified.  Does not read PyTorch
    
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
    x_min, x_max = 0, np.max(bulk_fit)
    
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
        
    return sigma


def plot_density(to_plot, sigma, Q, method="MP", color='blue', cutoff=0.0):
    """Method = 'MP' or 'QC'
    
    """

    if method == "MP":
        to_plot = np.sort(to_plot)
        x_min, x_max = 0, np.max(to_plot)
        x, y = marchenko_pastur_pdf(x_min, x_max, Q, sigma)
    elif method == "QC":
        cutoff = np.sqrt(cutoff)
        to_plot = np.sort(to_plot)
        x_min, x_max = 0, np.max(to_plot)
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


def matrix_soft_rank(W):
    """compute the matrix soft rank (or stable rank), given rectangular numpy matrix W""
    See:  https://arxiv.org/abs/1810.01075 """ 
    W = W / np.trace(W)
    u, sv, vh = svd(W)
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


def plot_loghist(x, bins, xmin):
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
    plt.savefig(figname)
    return 

#TODO:
# check alpha is decreasing
def fit_clipped_powerlaw(evals, xmin=None, verbose=False, min_alpha=5.0):
    """Fits a powerlaw only, not a truncate power law
       clips off the max evals until a powerlaw is found, or stops half-way into the ESD
       
       Used to remove power-law tail fingers, which may result from finite size effects
       
       Assumes evalsa re in sort order
       
       Does not allow alpha to increase; only activates if alpha < min_alpha """
       
    assert(evals[-1]> evals[0])  
    N = int(len(evals)/4)
    xmax = np.max(evals)
    prev_fit = powerlaw.Fit(evals, xmin=xmin, xmax=xmax, verbose=verbose)
    prev_alpha = prev_fit.alpha
    
    first_fit = prev_fit
    
    for idx in range(1,N):
        xmax = np.max(evals[-idx])
         
        fit = powerlaw.Fit(evals, xmin=xmin, xmax=xmax, verbose=verbose)
        print("fit alpha",fit.alpha)  
        
        if fit.alpha > prev_alpha:
            fit = prev_fit
            break
        
        if fit.alpha <= min_alpha: 
            print("stopping")  
            break
        
        # stop when distribtion becomes power law
        R, p = fit.distribution_compare('truncated_power_law', 'power_law', normalized_ratio=True)
        if R > 0.0:
            break
        
        prev_fit = fit
        prev_alpha = fit.alpha
            
    if idx == N:
        fit = first_fit
     
    return fit
             
        
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
    divergence = (sp.stats.entropy(p, m) + sp.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance
