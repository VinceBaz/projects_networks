# -*- coding: utf-8 -*-
"""
For testing projects_networks.assortativity functionality
"""

import numpy as np
from projects_networks import assortativity


def test_weighted_assort():

    A = np.array([[0, 1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])

    M = np.array([1, 1, 2, 2])

    assert assortativity.weighted_assort(A, M) == 1
