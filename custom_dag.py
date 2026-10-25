"""
Custom complex DAG
"""

import numpy as np

from common import N, bernoulli, dependence, descendant, xtheny

talent = bernoulli(0.2)
practice = bernoulli(np.where(talent == 1, 0.8, 0.2))


p_skill = np.zeros(N)
p_skill[(talent == 1) & (practice == 1)] = 0.99
p_skill[(talent == 1) & (practice == 0)] = 0.3
p_skill[(talent == 0) & (practice == 1)] = 0.5
p_skill[(talent == 0) & (practice == 0)] = 0.05

skill = bernoulli(p_skill)

selection = bernoulli(np.where(skill == 1, 0.7, 0.1))

p_result = np.zeros(N)
p_result[(selection == 1) & (skill == 1)] = 0.9
p_result[(selection == 1) & (skill == 0)] = 0.7
p_result[(selection == 0) & (skill == 1)] = 0.3
p_result[(selection == 0) & (skill == 0)] = 0.1

result = bernoulli(p_result)

print("Custom DAG Skill collider")
dependence(talent, practice, skill, ("talent", "practice",  "skill"))

print("Custom DAG Skill fork")
dependence(selection, skill, result, ("skill", "selection", "result"))

print("Custom DAG Result collider")
dependence(skill, selection, result, ("skill", "selection", "result"))

print("Custom DAG Ancestral sampling")
dependence(talent, skill, result, ("talent", "skill", "result"))

print("Custom DAG descendants")
descendant(practice, "practice")
descendant(result, "result")
descendant(skill, "skill")
descendant(selection, "selection")
xtheny(result, skill, ("result", "skill"))
xtheny(result, talent, ("result", "talent"))
xtheny(skill, talent, ("skill", "talent"))