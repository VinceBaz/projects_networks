# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:12:05 2020

@author: Vincent Bazinet
"""

import numpy as np
from scipy.spatial.distance import cdist
import bct
from mapalign.embed import compute_diffusion_map
import os
import abagen
from sklearn.decomposition import PCA 

def load_genes(Parcellation, data="lau", hemi="both", path=None):
    
    mainFolder = path+"/GeneExpression/"
    
    GENE = {}

    if hemi=="both":
        hemi=''
   
    #Gene names
    path = mainFolder+"GeneNames.npy"
    GENE["names"] = np.load(path, allow_pickle=True).tolist()
    GENE["names_brain"] = abagen.fetch_gene_group('brain')
    geneBrainsID = np.isin(GENE["names"], GENE["names_brain"])
    GENE["names_brain"] = np.array(GENE["names"])[geneBrainsID].tolist()

    #Gene expression
    path = mainFolder+"Gene_"+Parcellation+hemi+".npy"
    GENE["ex"] = np.load(path)
    GENE["ex_brain"] = GENE["ex"][:,geneBrainsID]
    
    #Principal components
    GENE["PCs"], GENE["PC_evs"] = getPCAgene(GENE["ex"].T)
    GENE["PCs_brain"], GENE["PC_evs_brain"] = getPCAgene(GENE["ex_brain"].T)
    
    return GENE

def load_annotations(Parcellation, data="lau", hemi="both", 
                     subset="all", path=None):

    mainFolder = path+"/brainNetworks/"+data+"/"

    ANN = {}
    
    if subset=="all":
        subset=''
    
    if hemi=="both":
        hemi=''
    
    #RSN
    path = mainFolder+"annotations/RSN"+Parcellation+hemi+".npy"
    RSN = np.load(path)
    ANN["RSN"] = RSN       
    path = mainFolder+"annotations/RSN_names.npy"
    ANN["RSN_names"] = np.load(path).tolist()
    
    #VonEconomo type
    path = mainFolder+"annotations/VE"+Parcellation+hemi+".npy"
    ANN["ve"] = np.load(path)
    path = mainFolder+"annotations/VE_names.npy"
    ANN["ve_names"] = np.load(path).tolist()
    
    #Spin tests
    path = mainFolder+"surrogates/spins_"+Parcellation+hemi+".npy"
    ANN["spin"] = np.load(path)
    
    #Morphometric properties
    if data=="HCP":
        path = mainFolder+"annotations/"+subset+"_T1wT2w_"+Parcellation+hemi+".npy"
        ANN["t1t2"] = np.mean(np.load(path), axis=0)
        path = mainFolder+"annotations/"+subset+"_Thi_"+Parcellation+hemi+".npy"
        ANN["thi"] = np.mean(np.load(path), axis=0)          
        
    return ANN
   
def load_network(Type, Parcellation, data="lau", hemi="both", binary=False, 
                 version=1, subset="all", communities=False, path=None):
    '''
    Function to load a network as well as its attributes
    ----------------------------------
    -> Type : Either SC or FC
    -> hemi : Either "both", "L" or "R"
    -> Parcellation : "68", "114", ... [if 'lau'] / "s400", "s800" [if "HCP"]
    -> version : either 1 (consensus computed without subcortex) or 2 
        (consensus conputed with subcortex)
    -> subset : either 'discov', 'valid' or 'all'
    -> path : path to the "data" folder in which the data will be stored. If
        none, then assumes that path is current folder.
    '''
    
    mainFolder = path+"/brainNetworks/"+data+"/"
    
    Network = {}

    if version==1:
        version=''
    else:
        version="_v2"
    
    if binary is True:
        binary="b"
    else:
        binary=''
    
    if subset=="all":
        subset=''
    
    if hemi=="both":
        Network["hemi"] = np.load(mainFolder+"matrices/"+subset+Type+Parcellation+binary+version+"/hemi.npy")
        hemi=''

    #Adjacency matrix [set negative values to 0, fill diagonal and make symmetric]
    path = mainFolder+"matrices/"+subset+Type+Parcellation+hemi+binary+version+".npy"
    A = np.load(path)
    A[A<0] = 0
    np.fill_diagonal(A,0)
    A = (A + A.T)/2
    Network["adj"] = A     
    #Coordinates
    path = mainFolder+"coords/coords"+Parcellation+hemi+".npy"
    coords = np.load(path)
    Network["coords"] = coords
    #Node strength
    Network["str"] = np.sum(A, axis=0)
    #Inverse of adjacency matrix
    inv = A.copy()
    inv[A>0] = 1/inv[A>0]
    Network["inv_adj"] = inv
    #distance
    Network["dist"] = cdist(coords, coords)
    #clustering coefficient
    Network["cc"] = bct.clustering_coef_wu(A)
    #shortest path
    path = mainFolder+"matrices/"+subset+Type+Parcellation+hemi+binary+version+"/sp.npy"
    sp = np.load(path)
    Network["sp"] = sp 
    #diffusion embedding
    de = compute_diffusion_map(A, n_components=10, return_result=True)
    Network["de"] = de[0]
    Network["de_extra"] = de[1]
    #Network["de"] = DiffusionMapEmbedding(alpha=0.5, diffusion_time=0, affinity='precomputed',
    #                       n_components=10).fit_transform(A)          
    #eigenvector centrality
    Network["ec"] = bct.eigenvector_centrality_und(A)
    #mean first passage time
    Network["mfpt"] = bct.mean_first_passage_time(A)
    #Betweenness centrality
    path = mainFolder+"matrices/"+subset+Type+Parcellation+hemi+binary+version+"/bc.npy"
    bc = np.load(path)
    Network["bc"] = bc
    #routing efficiency
    Network["r_eff"] = efficiency(Network)
    #diffusion efficiency
    Network["d_eff"] = efficiency_diffusion(Network)
    #Subgraph centrality
    Network["subc"] = bct.subgraph_centrality(Network["adj"])
    
    ###COMMUNITIES
    if communities is True:
        path = mainFolder+"matrices/"+subset+Type+Parcellation+hemi+binary+version+"/communities/"
        files = []
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path,i)) and 'ci_' in i:
                files.append(i)
        if len(files)>0:
            Network["ci"] = []
            for file in files:
                Network["ci"].append(np.load(os.path.join(path,file)))
            
            Network["ppc"] = []
            for i in range(len(files)):
                Network["ppc"].append(bct.participation_coef(A, Network["ci"][i]))
    
    if (data=="HCP") and (Type=="SC"):
        path = mainFolder+"matrices/"+subset+Type+Parcellation+hemi+"_lengths.npy"
        Network["lengths"] = np.load(path)
    
    return Network


def efficiency(Network):
    
    sp = Network["sp"]
    n = len(sp)
    
    efficiency_local= np.zeros((n))
    for i in range(n):
        efficiency_local[i] = np.mean(1/np.delete(sp[i,:],i))
    efficiency_global = np.mean(efficiency_local)
    
    return efficiency_global


def efficiency_diffusion(Network):
    
    if isinstance(Network, dict):
        mfpt = Network["mfpt"]
    else:
        mfpt = bct.mean_first_passage_time(Network)
    n = len(mfpt)
    
    efficiency_local= np.zeros((n))
    for i in range(n):
        efficiency_local[i] = np.mean(1/np.delete(mfpt[i,:],i))
    efficiency_global = np.mean(efficiency_local)
    
    return efficiency_global


def getPCAgene(genes, return_scores=False):
    '''
    Function to compute the Principal components of our 
    [genes x brain regions] matrix (see Burt et al., 2018).
    -----------
    INPUTS:
        genes -> gene expression data, [ndarray; shape:(n_nodes, n_genes)]
    OUTPUTS:
        PCs -> PCs of gene expression [ndarray; shape:(10, n_nodes)]
        ev  -> Explained variance ratio for PCs
    '''
    pca = PCA(n_components=10)
    scores = pca.fit_transform(genes)
    ev = pca.explained_variance_ratio_
    PCs = pca.components_
    
    if return_scores is False:
        return PCs, ev
    else:
        return PCs, ev, scores