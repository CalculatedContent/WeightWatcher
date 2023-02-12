import io

# for powerlaw warnings
import warnings
from contextlib import redirect_stdout, redirect_stderr

# remove warnings from powerlaw unless testing
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import powerlaw

from .constants import *

# when calling powerlaw methods,
# trap warnings, stdout and stderr
def pl_fit(data=None, xmin=None, xmax=None, verbose=False, distribution=POWER_LAW):
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

Fit = powerlaw.Fit
