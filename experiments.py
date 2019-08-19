'''
In this file, there are scripts that have been run to generate the results and
the data that are being visualized in experiments.ipynb. The results of these 
experiments have been stored in the notebook_experiments folder, which is not 
available on the repository (due to a lack of space). 

Get in touch with me if you want to access this folder! :)
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
Compute the local assortativity (Degree AND Random) and global assortativity
on the Erdos-Renyi Random Networks from 'SCRIPT #1'
'''

###SCRIPT #2.1 
#Generate random numbers
randNb = np.random.rand(100,200)
np.save("notebook_experiments/data/r2_rand_prop.npy", randNb)

r1 = np.load("notebook_experiments/data/r1_er100.npy")
randNb = np.load("notebook_experiments/data/r2_rand_prop.npy")

la_deg_er = np.zeros((100,200))
la_rand_er = np.zeros((100,200))
ga_deg_er = np.zeros((100))
ga_rand_er = np.zeros((100))
w_er = np.zeros((100,200,200))


for i in range(100):
    la_deg_er[i,:],w_er[i,:,:] = m.localAssort(r1[i,:,:], np.sum(r1[i,:,:], axis=0))
    la_rand_er[i,:],_ = m.localAssort(r1[i,:,:], randNb[i,:])
    ga_deg_er[i] = m.globalAssort(r1[i,:,:], np.sum(r1[i,:,:], axis=0))
    ga_rand_er[i] = m.globalAssort(r1[i,:,:], randNb[i,:])
    
np.save("notebook_experiments/results/r2_la_deg_er.npy", la_deg_er)
np.save("notebook_experiments/results/r2_la_rand_er.npy", la_rand_er)
np.save("notebook_experiments/results/r2_ga_deg_er.npy", ga_deg_er)
np.save("notebook_experiments/results/r2_ga_rand_er.npy", ga_rand_er)
np.save("notebook_experiments/data/r2_w_er.npy", w_er)

'''
SCRIPT #3
Generate 100 random Albert-Barabasi Networks,
with 200 nodes, and a density of 0.05
'''

ba100 = np.zeros((100,200,200))
for i in range(100):
    while True:
        #Generate the network
        ba100[i,:,:] = nx.to_numpy_array(nx.generators.random_graphs.barabasi_albert_graph(200, 5))
        #Make sure the network is fully connected
        R,_ = bct.breadthdist(ba100[i,:,:])
        if R[R==False].size==0:
            break    

np.save("notebook_experiments/data/r3_ba100.npy", ba100)

'''
SCRIPT #4
Compute the local assortativity (Degree AND Random) and global assortativity
on the Barabasi-Albert Random Networks from 'SCRIPT #3'
'''

ba100 = np.load("notebook_experiments/data/r3_ba100.npy")

la_deg_ba = np.zeros((100,200))
la_rand_ba = np.zeros((100,200))
ga_deg_ba = np.zeros((100))
ga_rand_ba = np.zeros((100))

#Load random numbers from SCRIPT #2
randNb = np.load("notebook_experiments/data/r2_rand_prop.npy")

for i in range(100):
    print(i)
    la_deg_ba[i,:],_ = m.localAssort(ba100[i,:,:], np.sum(ba100[i,:,:], axis=0))
    la_rand_ba[i,:],_ = m.localAssort(ba100[i,:,:], randNb[i,:])
    ga_deg_ba[i] = m.globalAssort(ba100[i,:,:], np.sum(ba100[i,:,:], axis=0))
    ga_rand_ba[i] = m.globalAssort(ba100[i,:,:], randNb[i,:])
    
np.save("notebook_experiments/results/r4_la_deg_ba.npy", la_deg_ba)
np.save("notebook_experiments/results/r4_la_rand_ba.npy", la_rand_ba)
np.save("notebook_experiments/results/r4_ga_deg_ba.npy", ga_deg_ba)
np.save("notebook_experiments/results/r4_ga_rand_ba.npy", ga_rand_ba)

'''
SCRIPT #5
Generate a random Erdos-Renyi Network,
with 1000 nodes, and a density of 0.05
'''

r5_er = nx.to_numpy_array(nx.generators.random_graphs.erdos_renyi_graph(1000, 0.05))
np.save("notebook_experiments/data/r5_er.npy", r5_er)

'''
SCRIPT #6
Compute the local assortativity (degree) of the network generated in script #5
'''

r5_er = np.load("notebook_experiments/data/r5_er.npy")
ga_deg_r5_er = m.globalAssort(r5_er, np.sum(r5_er, axis=0))
la_deg_r5_er,_ = m.localAssort(r5_er, np.sum(r5_er, axis=0))

np.save("notebook_experiments/results/r6_ga_deg_er.npy", ga_deg_r5_er)
np.save("notebook_experiments/results/r6_la_deg_er.npy", la_deg_r5_er)
