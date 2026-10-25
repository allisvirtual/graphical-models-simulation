"""
Student SAT and letter distribution
"""

import numpy as np
from common import N, bernoulli, dependence, rng, xtheny, descendant


difficulty = bernoulli(0.4)
intelligence = bernoulli(0.3)

p_grade = np.zeros((N, 3))

p_grade[(difficulty == 1) & (intelligence == 1)] = [0.5, 0.3, 0.2]
p_grade[(difficulty == 1) & (intelligence == 0)] = [0.05, 0.25, 0.7]
p_grade[(difficulty == 0) & (intelligence == 1)] = [0.9, 0.08, 0.02]
p_grade[(difficulty == 0) & (intelligence == 0)] = [0.3, 0.4, 0.3]

grade = np.array([
    rng.choice(3, p=probabilities)
    for probabilities in p_grade
])

SAT = bernoulli(np.where(intelligence == 1, 0.8, 0.05))

p_letter = np.zeros(N)

p_letter[grade == 0] = 0.9
p_letter[grade == 1] = 0.6
p_letter[grade == 2] = 0.01

letter = bernoulli(p_letter)

print("Student DAG")
dependence(difficulty, intelligence, letter, ("difficulty", "intelligence", "letter"))

print("SAT + difficulty -> letter")
dependence(SAT, difficulty, letter, ("SAT", "difficulty", "letter"))

xtheny(letter, difficulty, ("letter", "difficulty"))