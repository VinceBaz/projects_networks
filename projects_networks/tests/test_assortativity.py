# -*- coding: utf-8 -*-
"""
For testing projects_networks.assortativity functionality
"""

import numpy as np
from projects_networks.assortativity import weighted_assort
from projects_networks import assortativity

def test_weighted_assort():

    # binary
    A = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [1, 1, 0, 0],
                  [1, 1, 0, 0]])
    M = np.array([1, 1, 2, 2])
    N = np.array([2, 2, 1, 1])
    assert weighted_assort(A, M) == -1
    assert weighted_assort(A, M, N) == 1

    # weighted
    A = np.array([[0, 2, 1, 1],
                  [2, 0, 1, 1],
                  [1, 1, 0, 2],
                  [1, 1, 2, 0]])
    assert weighted_assort(A, M) == 0

    # directed
    A = np.array([[0, 1, 1, 0],
                  [1, 0, 1, 1],
                  [0, 1, 0, 0],
                  [1, 1, 1, 0]])
    M = np.array([1, 1, 1, 2])
    assert np.isclose(assortativity.weighted_assort(A, M, directed=True),
                      -0.25)


def test_global_assort():

    A = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [1, 1, 0, 0],
                  [1, 1, 0, 0]])

    M = np.array([1, 1, 2, 2])

    assert assortativity.global_assort(A, M) == -1
