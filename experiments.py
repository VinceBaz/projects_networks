'''
In this file, there are scripts that have been run to generate the results, as
well as some of the data that is being visualized in experiments.ipynb. The
results of these experiments have been stored in the notebook_experiments folder,
which is not available on the repository (due to a lack of space). Get in touch 
with me if you want to access this folder! :)
'''

'''
SCRIPT #0
Import Statements
'''

import networkx as nx
import numpy as np
import bct
from assortativity import measures as m

'''
SCRIPT #1
Generate 100 random Erdos-Renyi Networks,
with 200 nodes, and a density of 0.05
'''

r1 = np.zeros((100,200,200))
reachable = []
R = np.zeros((100,200,200))
for i in range(100):
    while True:
        #Generate the network
        r1[i,:,:] = nx.to_numpy_array(nx.generators.random_graphs.erdos_renyi_graph(200, 0.05))
        #Make sure the network is fully connected
        R,_ = bct.breadthdist(r1[i,:,:])
        if R[R==False].size==0:
            break
        
np.save("notebook_experiments/data/r1_er100.npy", r1)

'''
SCRIPT #2
Compute the local assortativity (Degree AND Random) on the Erdos-Renyi Random Networks
from 'SCRIPT #1'
'''

r1 = np.load("notebook_experiments/data/r1_er100.npy")

la_deg_er = np.zeros((100,200))
la_rand_er = np.zeros((100,200))

#Generate random numbers
randNb = np.random.rand(100,200)

for i in range(100):
    la_deg_er[i,:],_ = m.localAssort(r1[i,:,:], np.sum(r1[i,:,:], axis=0))
    la_rand_er[i,:],_ = m.localAssort(r1[i,:,:], randNb[i,:])
    
np.save("notebook_experiments/results/r2_la_deg_er.npy", la_deg_er)
np.save("notebook_experiments/results/r2_la_rand_er.npy", la_rand_er)