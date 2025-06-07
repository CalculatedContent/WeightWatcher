
'''
====================================================================================================================================================
This Python file defines a custom power law fitting class (WWFit) and functions for interacting with the powerlaw package. Let's break down the code:
1. Imports and Configuration
Imports: The code imports essential libraries:
io for string manipulation.
warnings to control warning handling.
contextlib for redirecting output streams.
sys for system-level settings.
powerlaw for power law fitting.
numpy for numerical operations.
constants for importing relevant constants from the WeightWatcher module.
supported_distributions: A dictionary mapping distribution names to their corresponding classes from the powerlaw package.
Logging: The code sets up a logger named WW_NAME (defined in the constants module).
2. WWFit Class
Purpose: This class provides a custom wrapper around the powerlaw.Fit class, simplifying the power law fitting process.
Constructor:
Initializes attributes like data, xmin, xmax, distribution, N, xmins.
Fits the data to the specified distribution (currently only supports POWER_LAW).
Methods:
fit_power_law: Fits the data to a power law distribution using a custom method for calculating alpha and D.
__getattr__: Allows accessing attributes from the powerlaw classes using the same syntax as powerlaw.Fit.
plot_pdf: Plots the probability density function (PDF) of the data.
plot_power_law_pdf: Plots the PDF of the fitted power law distribution.
distribution_compare: Compares the current distribution with another distribution using powerlaw.loglikelihood_ratio.
3. Helper Functions
pl_fit: A wrapper function that either uses the custom WWFit class for power law fitting or the powerlaw.Fit class for other distributions, depending on the pl_package argument. It also handles potential warnings and output redirection during the fitting process.
pl_compare: A wrapper function that compares two distributions using the powerlaw.Fit.distribution_compare method, handling warnings and output redirection.
4. Global Variable
Fit: Alias for the WWFit class, making it easier to use.
In summary, this file provides a custom power law fitting class and helper functions that streamline the power law analysis process while also managing warnings and output redirection. This approach makes the code more user-friendly and consistent with the powerlaw package's interface.
===============================================================================================================================================================================================================================================================================================
'''

import io

# for powerlaw warnings
import warnings
from contextlib import redirect_stdout, redirect_stderr

# remove warnings from powerlaw unless testing
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import powerlaw
import numpy as np

from .constants import *

supported_distributions = {
    'power_law':                powerlaw.Power_Law,
    'lognormal':                powerlaw.Lognormal,
    'exponential':              powerlaw.Exponential,
    'truncated_power_law':      powerlaw.Truncated_Power_Law,
    'stretched_exponential':    powerlaw.Stretched_Exponential,
    'lognormal_positive':       powerlaw.Lognormal_Positive,
}


import logging
logger = logging.getLogger(WW_NAME) 


class WWFit(object):
    def __init__(self, data, xmin=None, xmax=None, distribution=POWER_LAW):
        assert distribution in [POWER_LAW], distribution
        super(WWFit, self).__init__()

        data = np.asarray(np.sort(data), dtype=np.float64)
        def find_limit(data, x, default):
            if x is None: return default
            return np.argmin(np.abs(data - x))
        self.i_min = find_limit(data, xmin, 0)
        self.i_max = find_limit(data, xmax, len(data) - 1)
        self.xmin = data[self.i_min]
        self.xmax = data[self.i_max]

        self.data = data[self.i_min:self.i_max+1]
        self.N = len(self.data)
        self.xmins = self.data[:-1]
        self.distribution = distribution

        self.dists = {}
        if   self.distribution == POWER_LAW:
            self.fit_power_law()
            self.dists[POWER_LAW] = self

        i = np.argmin(self.Ds)
        self.xmin = self.xmins[i]
        self.alpha = self.alphas[i]
        self.sigma = self.sigmas[i]
        self.D = self.Ds[i]

        # powerlaw package does this, so we replicate it here.
        self.data = self.data[self.data >= self.xmin]

    def __str__(self):
        return f"WWFit({self.distribution} xmin: {self.xmin:0.04f}, alpha: {self.alpha:0.04f}, sigma: {self.sigma:0.04f}, data: {len(self.data)})"

    def fit_power_law(self):
        log_data    = np. log(self.data, dtype=np.float64)
        self.alphas = np.zeros(self.N-1, dtype=np.float64)
        self.Ds     = np. ones(self.N-1, dtype=np.float64)

        for i, xmin in enumerate(self.data[:-1]):
            n = float(self.N - i)
            alpha = 1 + n / (np.sum(log_data[i:]) - n * log_data[i])
            self.alphas[i] = alpha
            if alpha > 1:
                self.Ds[i] = np.max(np.abs(
                    1 - (self.data[i:] / xmin) ** (-alpha + 1) -    # Theoretical CDF\
                    np.arange(n) / n                                # Actual CDF
                ))

        self.sigmas = (self.alphas - 1) / np.sqrt(self.N - np.arange(self.N-1))

    def __getattr__(self, item):
        """ Needed for replicating the behavior of the powerlaw.Fit class"""
        if item in self.dists: return self.dists[item]
        raise AttributeError(item)

    def plot_pdf(self, **kwargs):
        """ Needed for replicating the behavior of the powerlaw.Fit class"""
        return powerlaw.plot_pdf(data=self.data, linear_bins=False, **kwargs)

    def plot_power_law_pdf(self, ax, **kwargs):
        """ Needed for replicating the behavior of the powerlaw.Power_Law class"""
        assert ax is not None

        # Formula taken directly from the powerlaw package.
        bins = np.unique(self.data)
        PDF = (bins ** -self.alpha) * (self.alpha-1) * (self.xmin**(self.alpha-1))

        assert np.min(PDF) > 0

        ax.plot(bins, PDF, **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")

    def distribution_compare(self, _dist1, _dist2, **kwargs):
        """
        Mimics the interface of a powerlaw.Fit object by passing through to powerlaw's functional API.
        """
        def get_loglikelihoods(_dist):
            if _dist in ["power_law"]:
                return np.log((self.data ** -self.alpha) * (self.alpha - 1) * self.xmin**(self.alpha-1))
            else:
                if _dist in self.dists: dist = self.dists[_dist]
                else:
                    dist = supported_distributions[_dist](
                        data = self.data, xmin=self.xmin, xmax=None, discrete=False, fit_method="Likelihood",
                        parameter_range=None, parent_Fit=None
                    )
                    self.dists[_dist] = dist

                return dist.loglikelihoods(self.data)

        return powerlaw.loglikelihood_ratio(
            get_loglikelihoods(_dist1),
            get_loglikelihoods(_dist2),
            nested=_dist1 in _dist2, **kwargs
        )

Fit = WWFit

# when calling powerlaw methods,
# trap warnings, stdout and stderr
def pl_fit(data=None, xmin=None, xmax=None, verbose=False, distribution=POWER_LAW, pl_package=WW_POWERLAW_PACKAGE):
    
    if xmax==FORCE:
        xmax=np.max(data)
    
    if pl_package==WW_POWERLAW_PACKAGE and distribution==POWER_LAW:
        logger.info("PL FIT running NEW power law method")
        return WWFit(data, xmin=xmin, xmax=xmax, distribution=distribution)
        
    else:
        
        logger.info(f"PL FIT running OLD power law method with  xmax={xmax}")
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f), warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            return powerlaw.Fit(data, xmin=xmin, xmax=xmax, verbose=verbose, distribution=distribution,
                                xmin_distribution=distribution)
            
            

def pl_compare(fit, dist):
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f), warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        return fit.distribution_compare(dist, TRUNCATED_POWER_LAW, normalized_ratio=True)
