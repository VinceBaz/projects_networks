# -*- coding: utf-8 -*-
"""
For testing projects_networks.diffusion functionality
"""

import numpy as np
from projects_networks import diffusion


def test_transition_matrix():

    A = np.array([[0, 2, 1, 1],
                  [2, 0, 1, 1],
                  [0, 1, 0, 3],
                  [2, 1, 1, 0]])

    T = np.array([[0, 0.5, 0.25, 0.25],
                  [0.5, 0, 0.25, 0.25],
                  [0, 0.25, 0, 0.75],
                  [0.5, 0.25, 0.25, 0]])

    assert np.all(diffusion.transition_matrix(A) == T)


def test_getPersoPR():

    A = np.array([[0, 1, 1, 1, 1, 0],
                  [1, 0, 1, 1, 1, 2],
                  [1, 1, 0, 1, 1, 1],
                  [1, 1, 1, 0, 1, 1],
                  [1, 1, 1, 1, 0, 1],
                  [0, 2, 1, 1, 1, 0]])

    perso = diffusion.getPersoPR(A, np.array([1]))

    sum_probabilities = perso[0, 0, :].sum()

    assert np.isclose(sum_probabilities, 1, rtol=1e-15, atol=1e-15)
