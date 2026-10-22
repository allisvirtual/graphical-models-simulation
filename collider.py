"""COLLIDER:  tech -> hired <- comm,  and the descendant  hired -> directory

Technical and communication skill are unrelated in the applicant pool.
The company hires on either, so 'hired' is a common effect of both.
"""

import numpy as np

from common import N, bernoulli, dependence

tech = bernoulli(0.5)
comm = bernoulli(0.5)

# P(hired | tech, comm), one row of the table per combination
p_hired = np.zeros(N)
p_hired[(tech == 1) & (comm == 1)] = 0.95
p_hired[(tech == 1) & (comm == 0)] = 0.50
p_hired[(tech == 0) & (comm == 1)] = 0.50
p_hired[(tech == 0) & (comm == 0)] = 0.05

hired = bernoulli(p_hired)

directory = bernoulli(np.where(hired == 1, 0.95, 0.02))

print("COLLIDER: tech -> hired <- comm")
dependence(tech, comm, hired, ("tech", "comm", "hired"))

print("COLLIDER: conditioning on the descendant instead")
dependence(tech, comm, directory, ("tech", "comm", "directory"))
