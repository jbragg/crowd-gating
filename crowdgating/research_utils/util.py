"""util.py

General utilities

"""

from __future__ import division
import os
import sys
import warnings

def get_or_default(d, k, default_v):
    warnings.warn("Don't use this function anymore. Use dict.get(), which does the same thing.",
                  DeprecationWarning)
    try:
        return d[k]
    except KeyError:
        return default_v

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 

def last_equal(lst):
    """Return longest sequence of equal elements at the end of a list.

    >>> last_equal([])
    []
    >>> last_equal([1, 2, 3, 3, 3])
    [3, 3, 3]
    >>> last_equal([1, 2, 3])
    [3]

    """
    els = []
    for v in reversed(lst):
        if len(els) == 0 or v == els[0]:
            els.append(v)
        else:
            break
    return els

def last_true(lst, f):
    """Return longest sequence of elements at end of list that pass f.

    >>> last_true([], lambda x: x % 2 == 1)
    []
    >>> last_true([1, 2, 3, 3, 3], lambda x: x % 2 == 1)
    [3, 3, 3]
    >>> last_true([1, 2, 3], lambda x: x % 2 == 1)
    [3]
    >>> last_true([1, 2, 3], lambda x: x % 2 == 0)
    []

    """
    els = last_equal(map(f, lst))
    if len(els) == 0 or els[0]:
        return lst[-1 * len(els):]
    else:
        return []

def midpoints(start, end, n):
    """Return midpoints of n even-distributed buckets in [start, end].

    >>> midpoints(0.5, 1, 2)
    [0.625, 0.875]
    >>> [round(x, 10) for x in midpoints(0.5, 1, 5)]
    [0.55, 0.65, 0.75, 0.85, 0.95]

    """
    import numpy as np
    endpoints = np.linspace(start, end, n + 1)
    return [(x+y)/2 for x, y in zip(endpoints[:-1], endpoints[1:])]


#--------- Statistics ------

from scipy.special import gamma
import scipy.stats as ss

beta = ss.beta.pdf

def dbeta(x, a, b):
    return gamma(a+b)/(gamma(a)*gamma(b)) * \
           ((a-1) * x**(a-2) * (1-x)**(b-1) - x**(a-1) * (b-1) * (1-x)**(b-2))

assert dbeta(0.5, 2, 2) == 0
assert dbeta(0.6, 2, 2) != 0

def dirichlet_mode(x):
    return [(v - 1) / (sum(x) - len(x)) for v in x]

def beta_fit(mode, mag):
    """Return parameters for beta distribution with given mode.

    Solves a system of linear equations to find alpha and beta s.t.
    alpha + beta = mag;
    (alpha - 1) / (alpha + beta - 2) = mode.

    Args:
        mode:  Mode of desired beta distribution.
        mag:   Magnitude of alpha + beta.

    Returns:
        x: Numpy array of [alpha, beta]

    >>> beta_fit(0.1, 7)
    array([ 1.5,  5.5])
    >>> beta_fit(0.2, 7)
    array([ 2.,  5.])
    >>> beta_fit(0.3, 7)
    array([ 2.5,  4.5])

    """
    import numpy as np
    a = np.array([[mode - 1, mode],
                  [1, 1]])
    b = np.array([2 * mode  - 1, mag])
    x = np.linalg.solve(a, b)
    return x

def truncnorm_sample(lower, upper, mu, std, size=1):
    """Sample from a truncated normal distribution.

    More intuitive version of scipy truncnorm function.

    Args:
        lower:  Lower bound.
        uppper: Upper bound.
        mu:     Mean.
        std:    Standard deviation.
        size:   Number of samples.

    Returns: Numpy array.

    """
    import numpy as np
    if std == 0:
        return np.array([mu for _ in xrange(size)])
    else:
        return ss.truncnorm.rvs((lower - mu) / std, (upper - mu) / std,
                                loc=mu, scale=std, size=size)


#-----------  Multiprocessing -------

import traceback
import multiprocessing as mp
import re

def init_worker():
    """Function to make sure everyone happily exits on KeyboardInterrupt

    See https://stackoverflow.com/questions/1408356/
    keyboard-interrupts-with-pythons-multiprocessing-pool

    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def run_functor(functor, x):
    """
    Given a no-argument functor, run it and return its result. We can
    use this with multiprocessing.map and map it over a list of job
    functors to do them.

    Handles getting more than multiprocessing's pitiful exception output

    https://stackoverflow.com/questions/6126007/
    python-getting-a-traceback-from-a-multiprocessing-process

    """
    try:
        # This is where you do your actual work
        return functor(x)
    except:
        # Put all exception text into an exception and raise that
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))

def cpu_count():
    """Return number of cpus respecting numactl restrictions.

    Never exceeds number of cpus specified by mp.cpu_count().

    Based on code from https://stackoverflow.com/questions/1006289/
    how-to-find-out-the-number-of-cpus-using-python.

    """
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return min(res, mp.cpu_count())
        return mp.cpu_count()
    except IOError:
        return mp.cpu_count()


#-----------  I/O ------------------

def save_object(obj, filename):
    """
    Uses cPickle module to save the object to specified file name
    """
    import cPickle as pickle
    f = file(filename, 'wb')
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()   

def load_object(filename):
    """
    Uses the cPickle module to load a saved object
    """
    import cPickle as pickle
    f = file(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

#------------- Sorting ---------------
def get_top(obj_dict, max_count):
    """
    Sorts the keys in a dictionary based on the values and 
    returns the top max_count keys as a list. Useful in implementing search 
    algorithms
    """
    sorted_list = sorted(obj_dict.keys(), key=obj_dict.get, reverse=True)
    if len(obj_dict) <= max_count:
        return sorted_list
    else:
        return sorted_list[0:max_count]
