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
        data = self.data[self.data >= self.xmin]
        return powerlaw.plot_pdf(data=data, linear_bins=False, **kwargs)

    def plot_power_law_pdf(self, ax, **kwargs):
        """ Needed for replicating the behavior of the powerlaw.Power_Law class"""
        assert ax is not None

        # Formula taken directly from the powerlaw package.
        data = self.data[self.data >= self.xmin]
        bins = np.unique(data)
        PDF = (data ** -self.alpha) * (self.alpha-1) * (self.xmin**(self.alpha-1))

        assert np.min(PDF) > 0

        ax.plot(bins, PDF, **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")

    def distribution_compare(self, _dist1, _dist2, **kwargs):
        """
        Mimics the interface of a powerlaw.Fit object by passing through to powerlaw's functional API.
        """
        data = self.data[self.data >= self.xmin]
        def get_loglikelihoods(_dist):
            if _dist in ["power_law"]:
                return np.log((data ** -self.alpha) * (self.alpha - 1) * self.xmin**(self.alpha-1))
            else:
                if _dist in self.dists: dist = self.dists[_dist]
                else:
                    dist = supported_distributions[_dist](
                        data = data, xmin=self.xmin, xmax=None, discrete=False, fit_method="Likelihood",
                        parameter_range=None, parent_Fit=None
                    )
                    self.dists[_dist] = dist

                return dist.loglikelihoods(data)

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
