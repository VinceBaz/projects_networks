'''
In this file, there are scripts that have been run to generate the results and
the data that are being visualized in experiments.ipynb. The results of these 
experiments have been stored in the notebook_experiments folder, which is not 
available on the repository (due to a lack of space). 

As soon as I will have a sufficient number of interesting experiment,
I will definively have publish my data on something like figshare.
In the meanwhile, get in touch with me if you want to access this folder! :)

Author: Vincent Baznet

'''

'''
SCRIPT #0
Import Statements
'''

import networkx as nx
import numpy as np
import bct
from assortativity import measures as m
from assortativity import tools
import random

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

###SCRIPT #2.2
#Compute local assortativity, but always using the same rand distribution of attributes
la_samerand_er = np.zeros((100,200))
for i in range(100):
    la_samerand_er[i,:],_ = m.localAssort(r1[i,:,:], randNb[0,:])
    
np.save("notebook_experiments/results/r2_la_samerand_er.npy", la_samerand_er)

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
Generate a single random Erdos-Renyi Network,
with 400 nodes, and a density of 0.05
'''

r5_er = nx.to_numpy_array(nx.generators.random_graphs.erdos_renyi_graph(400, 0.05))
r5_randNb = np.random.rand(400)

r5_logspace = np.logspace(-3,0,50)
r5_logspace = 1-r5_logspace
r5_logspace = np.delete(r5_logspace, 49)

np.save("notebook_experiments/data/r5_er.npy", r5_er)
np.save("notebook_experiments/data/r5_randNb.npy", r5_randNb)
np.save("notebook_experiments/data/r5_logspace.npy", r5_logspace)

'''
SCRIPT #6
Compute the local assortativity (degree and random) of the network generated 
in script #5, using different techniques and alphas, and store extra
information about the measure computed (means and std of the distributions)
'''

r5_randNb = np.load("notebook_experiments/data/r5_randNb.npy")
r5_er = np.load("notebook_experiments/data/r5_er.npy")
r5_logspace = np.load("notebook_experiments/data/r5_logspace.npy")

deg = np.sum(r5_er, axis=0)

#Global Assortativity
r6_ga_deg_er = m.globalAssort(r5_er, np.sum(r5_er, axis=0))
r6_ga_rand_er = m.globalAssort(r5_er, r5_randNb)

#Local Assortativity
r6_w_deg_er = np.zeros((50,400,400))

r6_la_deg_er_peel = np.zeros((50,400))
r6_la_deg_er_w = np.zeros((50,400))
r6_la_deg_er_w_t = np.zeros((50,400))

r6_la_deg_er_peel_extra = np.zeros((50), dtype=object)
r6_la_deg_er_w_extra = np.zeros((50), dtype=object)
r6_la_deg_er_w_t_extra = np.zeros((50), dtype=object)

r6_la_rand_er_peel = np.zeros((50,400))
r6_la_rand_er_w = np.zeros((50,400))
r6_la_rand_er_w_t = np.zeros((50,400))

r6_la_rand_er_peel_extra = np.zeros((50), dtype=object)
r6_la_rand_er_w_extra = np.zeros((50), dtype=object)
r6_la_rand_er_w_t_extra = np.zeros((50), dtype=object)

count=0
for i in r5_logspace:
    
    print(count)
    
    r6_la_deg_er_peel[count,:], r6_w_deg_er[count,:,:], r6_la_deg_er_peel_extra[count]  = m.localAssort(r5_er, deg, pr=i, method="Peel", thorndike=False, return_extra=True)
    r6_la_deg_er_w[count,:],_, r6_la_deg_er_w_extra[count] = m.localAssort(r5_er, deg, pr=i, method="weighted", thorndike=False, return_extra=True)
    r6_la_deg_er_w_t[count,:],_, r6_la_deg_er_w_t_extra[count] = m.localAssort(r5_er, deg, pr=i, method="weighted", thorndike=True, return_extra=True)
    
    r6_la_rand_er_peel[count,:],_, r6_la_rand_er_peel_extra[count] = m.localAssort(r5_er, r5_randNb, pr=i, method="Peel", thorndike=False, return_extra=True)
    r6_la_rand_er_w[count,:],_, r6_la_rand_er_w_extra[count] = m.localAssort(r5_er, r5_randNb, pr=i, method="weighted", thorndike=False, return_extra=True)
    r6_la_rand_er_w_t[count,:],_, r6_la_rand_er_w_t_extra[count] = m.localAssort(r5_er, r5_randNb, pr=i, method="weighted", thorndike=True, return_extra=True)
    count+=1
    
r6_la_deg_er_peel[count,:], r6_w_deg_er[count,:,:], r6_la_deg_er_peel_extra[count] = m.localAssort(r5_er, deg, pr="multiscale", method="Peel", thorndike=False, return_extra=True)
r6_la_deg_er_w[count,:],_, r6_la_deg_er_w_extra[count] = m.localAssort(r5_er, deg, pr="multiscale", method="weighted", thorndike=False, return_extra=True)
r6_la_deg_er_w_t[count,:],_, r6_la_deg_er_w_t_extra[count] = m.localAssort(r5_er, deg, pr="multiscale", method="weighted", thorndike=True, return_extra=True)

r6_la_rand_er_peel[count,:],_, r6_la_rand_er_peel_extra[count] = m.localAssort(r5_er, r5_randNb, pr="multiscale", method="Peel", thorndike=False, return_extra=True)
r6_la_rand_er_w[count,:],_, r6_la_rand_er_w_extra[count] = m.localAssort(r5_er, r5_randNb, pr="multiscale", method="weighted", thorndike=False, return_extra=True)
r6_la_rand_er_w_t[count,:],_, r6_la_rand_er_w_t_extra[count] = m.localAssort(r5_er, r5_randNb, pr="multiscale", method="weighted", thorndike=True, return_extra=True)


np.save("notebook_experiments/results/r6_ga_deg_er.npy", r6_ga_deg_er)
np.save("notebook_experiments/results/r6_ga_rand_er.npy", r6_ga_rand_er)

np.save("notebook_experiments/results/r6_w_deg_er.npy", r6_w_deg_er)

np.save("notebook_experiments/results/r6_la_deg_er_peel.npy", r6_la_deg_er_peel)
np.save("notebook_experiments/results/r6_la_deg_er_w.npy", r6_la_deg_er_w)
np.save("notebook_experiments/results/r6_la_deg_er_w_t.npy", r6_la_deg_er_w_t)

np.save("notebook_experiments/results/r6_la_deg_er_peel_extra.npy", r6_la_deg_er_peel_extra)
np.save("notebook_experiments/results/r6_la_deg_er_w_extra.npy", r6_la_deg_er_w_extra)
np.save("notebook_experiments/results/r6_la_deg_er_w_t_extra.npy", r6_la_deg_er_w_t_extra)

np.save("notebook_experiments/results/r6_la_rand_er_peel.npy", r6_la_rand_er_peel)
np.save("notebook_experiments/results/r6_la_rand_er_w.npy", r6_la_rand_er_w)
np.save("notebook_experiments/results/r6_la_rand_er_w_t.npy", r6_la_rand_er_w_t)

np.save("notebook_experiments/results/r6_la_rand_er_peel_extra.npy", r6_la_rand_er_peel_extra)
np.save("notebook_experiments/results/r6_la_rand_er_w_extra.npy", r6_la_rand_er_w_extra)
np.save("notebook_experiments/results/r6_la_rand_er_w_t_extra.npy", r6_la_rand_er_w_t_extra)

'''
SCRIPT #7
Compute the local assortativity (degree) of the ER random networks generated in
SCRIPT#1, for increasing values of alpha.
'''

r1 = np.load("notebook_experiments/data/r1_er100.npy")

r7_la_pr_er = np.zeros((100,50,200))
r7_w_pr_er = np.zeros((100,50,200,200))

for i in range(100):
    count=0
    print(i)
    for j in np.arange(0.02,1.01,0.02):
        r7_la_pr_er[i,count,:],r7_w_pr_er[i,count,:,:] = m.localAssort(r1[i,:,:], np.sum(r1[i,:,:], axis=0), pr=j)
        count+=1
        
np.save("notebook_experiments/results/r7_la_pr_er.npy", r7_la_pr_er)
np.save("notebook_experiments/results/r7_w_pr_er.npy", r7_w_pr_er)

'''
SCRIPT #8
'''

r8_er = nx.to_numpy_array(nx.generators.random_graphs.erdos_renyi_graph(200, 0.1))
np.save("notebook_experiments/data/r8_er.npy", r8_er)

'''
SCRIPT #9
Generate a distribution of attribute in an ER network such that the standard 
deviation of the kurtosis of the weighted distribution of the attribute
values in all of the local neighborhood is smaller than expected.
'''

r9_randNb = np.random.rand(200)

r9_la_kurt = np.zeros((50,200))
r9_randNb_kurt = np.zeros((50,200))
r9_la_extra_kurt = np.zeros((50), dtype=object)

r9_randNb_kurt[0,:] = r9_randNb
r9_la_kurt[0,:], _, r9_la_extra_kurt[0] = m.localAssort(r8_er, r9_randNb, pr="multiscale", method="weighted", thorndike=False, return_extra=True)

old_kurt = np.zeros((200)) 
for i in range(200):
    old_kurt[i] = r9_la_extra_kurt[0][i]["x_kurt"]
old_kurt_std = np.std(old_kurt)
og_std = old_kurt_std
c1=0
c2=1
while c2<500:
    
    print("c1: ",c1," - c2: ",c2)
    
    x=0
    y=0
    while x==y:
        x = random.randint(0,199)
        y = random.randint(0,199)
    
    r9_randNb[x], r9_randNb[y] = r9_randNb[y], r9_randNb[x]
    
    r9_la,_, r9_la_extra = m.localAssort(r8_er, r9_randNb, pr="multiscale", method="weighted", thorndike=False, return_extra=True)
    c1+=1
    
    kurt = np.zeros((200))
    for i in range(200):
        kurt[i] = r9_la_extra[i]["x_kurt"]
    kurt_std = np.std(kurt)
    
    if kurt_std < old_kurt_std:
        
        if c2%10==0:
            r9_la_kurt[int(c2/10),:] = r9_la
            r9_la_extra_kurt[int(c2/10)] = r9_la_extra
            r9_randNb_kurt[int(c2/10),:] = r9_randNb
        c2+=1
        old_kurt_std = kurt_std
    else:
        r9_randNb[x], r9_randNb[y] = r9_randNb[y], r9_randNb[x]
        
np.save("notebook_experiments/results/r9_la_kurt.npy", r9_la_kurt)
np.save("notebook_experiments/results/r9_randNb_kurt.npy", r9_randNb_kurt)
np.save("notebook_experiments/results/r9_la_extra_kurt.npy", r9_la_extra_kurt)

'''
SCRIPT #10
Generate a network with a random distribution of attribute values
such that the standard  deviation of the kurtosis of the weighted 
distribution of these attribute values in all of the local neighborhood 
is smaller than expected by chance.
'''

PrevG = nx.to_numpy_array(nx.generators.random_graphs.erdos_renyi_graph(200, 0.1))

r10_randNb = np.random.rand(200)

r10_la_kurt = np.zeros((50,200))
r10_la_extra_kurt = np.zeros((50), dtype=object)

r10_la_kurt[0,:], _, r10_la_extra_kurt[0] = m.localAssort(PrevG, r10_randNb, pr="multiscale", method="weighted", thorndike=False, return_extra=True)

old_kurt = np.zeros((200)) 
for i in range(200):
    old_kurt[i] = r10_la_extra_kurt[0][i]["x_kurt"]
old_kurt_std = np.std(old_kurt)
og_std = old_kurt_std
c1=0
c2=1
while c2<500:
    
    print("c1: ",c1," - c2: ",c2," - SD kurt: ",old_kurt_std)
    
    edges = np.where(PrevG==1)
    noEdges = np.where(PrevG==0)
    
    #Choose edge + no edge at random
    rand1 = random.randrange(0,len(edges[0]))
    rand2 = random.randrange(0,len(edges[0]))
    e1 = [edges[0][rand1], edges[1][rand1]]
    e2 = [noEdges[0][rand2], noEdges[1][rand2]]   

    #swap edges
    newG = PrevG.copy()
    newG[e1[0],e1[1]] = 0
    newG[e2[0], e2[1]] = 1
        
    r10_la,_, r10_la_extra = m.localAssort(newG, r10_randNb, pr="multiscale", method="weighted", thorndike=False, return_extra=True)
    c1+=1
    
    kurt = np.zeros((200))
    for i in range(200):
        kurt[i] = r10_la_extra[i]["x_kurt"]
    kurt_std = np.std(kurt)
    
    if kurt_std < old_kurt_std:
        
        if c2%10==0:
            r10_la_kurt[int(c2/10),:] = r10_la
            r10_la_extra_kurt[int(c2/10)] = r10_la_extra
        c2+=1
        
        old_kurt_std = kurt_std
        PrevG = newG
        
    else:
        newG[e1[0],e1[1]] = 1
        newG[e2[0], e2[1]] = 0
        
np.save("notebook_experiments/results/r10_la_kurt.npy", r10_la_kurt)
np.save("notebook_experiments/results/r10_newG.npy", newG)
np.save("notebook_experiments/results/r10_la_extra_kurt.npy", r10_la_kurt)

'''
SCRIPT #11
Generate a random Erdos-Renyi network with 1000 nodes, and a density of 0.05
'''

r11_er = nx.to_numpy_array(nx.generators.random_graphs.erdos_renyi_graph(1000, 0.05))
r11_randNb = np.random.rand(1000)

r11_la_p,_ = m.localAssort(r11_er, r11_randNb, method="Peel", thorndike=False)
r11_la_w,w,r11_la_extra_w = m.localAssort(r11_er, r11_randNb, method="weighted", thorndike=False, return_extra=True)

np.save("notebook_experiments/data/r11_er.npy", r11_er)
np.save("notebook_experiments/data/r11_randNb.npy", r11_randNb)
np.save("notebook_experiments/results/r11_la_p.npy", r11_la_p)
np.save("notebook_experiments/results/r11_la_w.npy", r11_la_w)
np.save("notebook_experiments/results/r11_la_extra_w.npy", r11_la_extra_w)

'''
SCRIPT #12.1
Generate 200 random Erdos-Renyi networks with 300 nodes and a density of 0.05
'''

r12 = np.zeros((200,300,300))
for i in range(200):
    
    print("Generate Network - ",i)
    
    while True:
        #Generate the network
        r12[i,:,:] = nx.to_numpy_array(nx.generators.random_graphs.erdos_renyi_graph(300, 0.05))
        #Make sure the network is fully connected
        R,_ = bct.breadthdist(r12[i,:,:])
        if R[R==False].size==0:
            break
        
r12_randNb = np.random.rand(300)

np.save("notebook_experiments/results/r12.npy", r12)
np.save("notebook_experiments/results/r12_randNb.npy", r12_randNb)

'''
SCRIPT #12.2
Compute the local assortatvity of these random networks
'''

r12 = np.load("notebook_experiments/results/r12.npy")
r12_randNb = np.load("notebook_experiments/results/r12_randNb.npy")

r12_la = np.zeros((200,300))
r12_la[0,:], w = m.localAssort(r12[0,:,:], r12_randNb)

for i in range(1,200):
    
    print("Compute Local Ass - ",i)
    
    r12_la[i,:],_ = m.localAssort(r12[i,:,:], r12_randNb, weights=w)
    
np.save("notebook_experiments/results/r12_la.npy", r12_la)

'''
#SCRIPT #13
Use the ER networks from script #12, and randomly swap edge 
to get random networks that have a global assortativity of 0.20
'''

r12_randNb = np.load("notebook_experiments/results/r12_randNb.npy")
r12 = np.load("notebook_experiments/results/r12.npy")


r13 = r12.copy()
for i in range(200):
    r13[i,:,:] = tools.assort_preserv_swap(r13[i,:,:], r12_randNb, 0.20)[0]
np.save("notebook_experiments/data/r13_erass.npy", r13)

'''
#SCRIPT #13.2
Compute the local assortativity on these networks
'''

r13 = np.load("notebook_experiments/data/r13_erass.npy")
r12_randNb = np.load("notebook_experiments/results/r12_randNb.npy")

r13_la = np.zeros((200,300))

for i in range(200):
    r13_la[i,:],_ = m.localAssort(r13[i,:,:], r12_randNb)

np.save("notebook_experiments/results/r13_la.npy", r13_la)
