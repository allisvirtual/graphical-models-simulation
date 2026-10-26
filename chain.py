"""CHAIN:  deploy -> bug -> tickets"""

import numpy as np

from common import bernoulli, dependence

deploy = bernoulli(0.3)
bug = bernoulli(np.where(deploy == 1, 0.6, 0.1))
tickets = bernoulli(np.where(bug == 1, 0.8, 0.15))

print("CHAIN: deploy -> bug -> tickets")
dependence(deploy, tickets, bug, ("deploy", "tickets", "bug"))
