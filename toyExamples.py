#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:10:45 2019

@author: gshafiei
"""

import numpy as np
from assortativity import measures as m
import matplotlib.pyplot as plt
import bct

M = np.array([1,2,3,4,5,6,7,8])

A = np.array([[0,1,0,0,0,0,0,0],
              [1,0,1,0,0,0,0,0],
              [0,1,0,1,0,0,0,0],
              [0,0,1,0,1,0,0,0],
              [0,0,0,1,0,1,0,0],
              [0,0,0,0,1,0,1,0],
              [0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,1,0]])

B = np.array([[0,0,1,0,0,0,0,0],
              [0,0,1,0,0,0,0,0],
              [1,1,0,1,0,0,0,0],
              [0,0,1,0,1,0,0,0],
              [0,0,0,1,0,1,0,0],
              [0,0,0,0,1,0,1,0],
              [0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,1,0]])

print(m.globalAssort(A,M))
print(m.globalAssort(B,M))

laAw,_ = m.localAssort(A,M, method="weighted")
laAp,_ = m.localAssort(A,M, method="Peel")

laBw,_ = m.localAssort(B,M, method="weighted")
laBp,_ = m.localAssort(B,M, method="Peel")

plt.figure()
plt.scatter(laAp, laBp, c=M)
plt.colorbar()
plt.plot([0,2], [0,2], linestyle="dashed", c="black")

plt.figure()
plt.scatter(laAw, laBw, c=M)
plt.colorbar()
plt.plot([0,1], [0,1], linestyle="dashed", c="black")

from networkx.generators.community import stochastic_block_model as sbm
import networkx as nx

C = nx.to_numpy_array(sbm([100, 100, 100, 100],
                          [[0.5, 0.01, 0.01, 0.01],
                          [0.01, 0.5, 0.01, 0.01],
                          [0.01, 0.01, 0.5, 0.01],
                          [0.01, 0.01, 0.01, 0.5]]))

D = nx.to_numpy_array(sbm([100, 100, 100, 100],
                          [[0.01, 0.5, 0.01, 0.01],
                          [0.5, 0.01, 0.01, 0.01],
                          [0.01, 0.01, 0.5, 0.01],
                          [0.01, 0.01, 0.01, 0.5]]))

N1 = np.random.rand((100))+1
N2 = np.random.rand((100))+2
N3 = np.random.rand((100))+3
N4 = np.random.rand((100))+4

N = np.concatenate((N1,N2, N3, N4))

laCw,_ = m.localAssort(C,N, method="weighted", thorndike=False)
laCp,_ = m.localAssort(C,N, method="Peel", thorndike=False)

laDw,_ = m.localAssort(D,N, method="weighted", thorndike=False)
laDp,_ = m.localAssort(D,N, method="Peel", thorndike=False)


plt.figure()
C2 = N[:,np.newaxis] * C
plt.imshow(C2)
plt.title("C")
plt.colorbar()

plt.figure()
D2 = N[:,np.newaxis] * D
plt.imshow(D2)
plt.title("D")
plt.colorbar()

plt.figure()
plt.scatter(np.arange(0,400),laCw)

plt.figure()
plt.scatter(np.arange(0,400),laCp)

plt.figure()
plt.scatter(np.arange(0,400),laDw)

plt.figure()
plt.scatter(np.arange(0,400),laDp)