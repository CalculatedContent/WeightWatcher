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

SMALL_N_CUTOFF = 20

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
        if self.N < SMALL_N_CUTOFF:
            print("SMALL N PL FIT")
            logger.info("SMALL N PL FIT")
            self.fit_powerlaw_smallN()
            return 
        
        return self.fit_power_law_standard()

    def fit_power_law_standard(self):
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
        
            
    
    def fit_powerlaw_smallN(self, k_min: int = 8, lambda_prior: float = 0.0):
        """
        Small-N continuous power-law fit:

          - Bias-corrected MLE: alpha_bc = 1 + (n - 1) / sum_j log(x_j / xmin)
          - Objective for xmin selection:
                J = D_ks - 0.868 / sqrt(n_tail) + lambda_prior * prior_pen
            where prior_pen = (alpha_bc - 2)^2  (ultra-local prior, off if lambda_prior=0)

        No trace-log gate, no eigenvalue rescaling, no lock-to-2.
        """

        log_data = np.log(self.data, dtype=np.float64)

        # Arrays similar to fit_power_law
        self.alphas = np.zeros(self.N - 1, dtype=np.float64)
        self.Ds     = np.ones(self.N - 1, dtype=np.float64)
        # Objective values (for internal selection)
        self.Js     = np.full(self.N - 1, np.inf, dtype=np.float64)

        for i, xmin in enumerate(self.data[:-1]):
            n_int = self.N - i        # tail size as int
            if n_int < k_min:
                continue
            n = float(n_int)

            # sum_j log(x_j / xmin) for j >= i
            s = np.sum(log_data[i:]) - n * log_data[i]
            if s <= 1e-12:
                # degenerate tail; skip
                continue

            # --- bias-corrected MLE (n-1 correction) ---
            alpha_bc = 1.0 + (n - 1.0) / s
            self.alphas[i] = alpha_bc

            if alpha_bc <= 1.0:
                # invalid exponent for continuous power law; skip
                continue

            # Tail data for this xmin
            tail = self.data[i:]

            # Theoretical CDF for continuous power law on [xmin, ∞):
            # F_fit(x) = 1 - (x/xmin)^(1 - alpha), x >= xmin
            F_fit = 1.0 - (tail / xmin) ** (1.0 - alpha_bc)

            # Empirical CDF: 0, 1/n, ..., (n-1)/n  (matches your original style)
            F_emp = np.arange(n_int, dtype=np.float64) / n
            Dks = float(np.max(np.abs(F_emp - F_fit)))
            self.Ds[i] = Dks

            # --- Objective 1A: KS-scaled tail-size encouragement ---
            prior_pen = (alpha_bc - 2.0) ** 2   # ultra-local prior (if lambda_prior > 0)
            J = Dks - 0.868 / np.sqrt(n) + lambda_prior * prior_pen
            self.Js[i] = J

        # Sigma like the original code (for reporting)
        self.sigmas = (self.alphas - 1.0) / np.sqrt(self.N - np.arange(self.N - 1))

        # ----- Choose best xmin by J; no fallback to fit_power_law -----
        if np.isfinite(self.Js).any():
            j_best = int(np.nanargmin(self.Js))
        else:
            # If k_min was too strict and no candidate survived, use all data as tail (i=0)
            j_best = 0
            xmin = self.data[0]
            n_int = self.N
            n = float(n_int)
            s = np.sum(log_data) - n * log_data[0]
            if s <= 1e-12:
                # pathological case; keep trivial defaults
                self.xmin  = xmin
                self.alpha = 1.0
                self.sigma = 0.0
                self.D     = 1.0
                self.data  = self.data[self.data >= self.xmin]
                return

            alpha_bc = 1.0 + (n - 1.0) / s
            self.alphas[j_best] = alpha_bc

            tail = self.data
            F_fit = 1.0 - (tail / xmin) ** (1.0 - alpha_bc)
            F_emp = np.arange(n_int, dtype=np.float64) / n
            Dks = float(np.max(np.abs(F_emp - F_fit)))
            self.Ds[j_best] = Dks

            prior_pen = (alpha_bc - 2.0) ** 2
            self.Js[j_best] = Dks - 0.868 / np.sqrt(n) + lambda_prior * prior_pen

        # Commit winner (similar to what __init__ does after fit_power_law)
        self.xmin  = self.data[j_best]
        self.alpha = self.alphas[j_best]
        self.sigma = self.sigmas[j_best]
        self.D     = self.Ds[j_best]

        # Match powerlaw package behavior: restrict data to data >= xmin
        self.data = self.data[self.data >= self.xmin]

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
