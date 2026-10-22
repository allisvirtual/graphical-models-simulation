"""FORK:  slow <- load -> error

High traffic makes responses slow and makes requests fail.
Slowness does not cause errors - they only share a parent.
"""

import numpy as np

from common import bernoulli, dependence

load = bernoulli(0.4)
slow = bernoulli(np.where(load == 1, 0.8, 0.1))
error = bernoulli(np.where(load == 1, 0.7, 0.05))

print("FORK: slow <- load -> error")
dependence(slow, error, load, ("slow", "error", "load"))
