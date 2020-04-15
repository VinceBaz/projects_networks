# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:12:05 2020

Code to load network-related data from a "data" folder stored locally
into an easy to use python dictionary.

@author: Vincent Bazinet
"""

import numpy as np
from scipy.spatial.distance import cdist
import bct
from mapalign.embed import compute_diffusion_map
import os
import abagen
from sklearn.decomposition import PCA


def load_genes(parcel, data="lau", hemi="both", path=None):
    '''
    Load gene-related information
    '''
    mainPath = path+"/GeneExpression/"

    GENE = {}

    if hemi == "both":
        hemi = ''

    # Gene names
    path = mainPath+"GeneNames.npy"
    if os.path.exists(path):
        # (all)
        GENE["names"] = np.load(path, allow_pickle=True).tolist()
        # (brain)
        GENE["names_brain"] = abagen.fetch_gene_group('brain')
        geneBrainsID = np.isin(GENE["names"], GENE["names_brain"])
        GENE["names_brain"] = np.array(GENE["names"])[geneBrainsID].tolist()

    # Gene expression
    path = mainPath+"Gene_"+parcel+hemi+".npy"
    if os.path.exists(path):
        GENE["ex"] = np.load(path)
        GENE["ex_brain"] = GENE["ex"][:, geneBrainsID]

    # Principal components
    GENE["PCs"], GENE["PC_evs"] = getPCAgene(GENE["ex"].T)
    GENE["PCs_brain"], GENE["PC_evs_brain"] = getPCAgene(GENE["ex_brain"].T)

    return GENE


def load_annotations(parcel, data="lau", hemi="both",
                     subset="all", path=None):
    '''
    Load non-topological annotations as well as some tools useful
    for investigating these annotations (e.g. spin samples)
    '''

    mainPath = path+"/brainNetworks/"+data+"/"

    ANN = {}

    if subset == "all":
        subset = ''

    if hemi == "both":
        hemi = ''

    # RSN
    path = mainPath+"annotations/RSN"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["RSN"] = np.load(path)

    # RSN names
    path = mainPath+"annotations/RSN_names.npy"
    if os.path.exists(path):
        ANN["RSN_names"] = np.load(path).tolist()

    # VonEconomo type
    path = mainPath+"annotations/VE"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["ve"] = np.load(path)

    # Von Economo names
    path = mainPath+"annotations/VE_names.npy"
    if os.path.exists(path):
        ANN["ve_names"] = np.load(path).tolist()

    # Spin tests
    path = mainPath+"surrogates/spins_"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["spin"] = np.load(path)

    # Morphometric property (T1w/T2w)
    path = mainPath+"annotations/"+subset+"_T1wT2w_"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["t1t2"] = np.mean(np.load(path), axis=0)

    # Morphometric property (Thickness)
    path = mainPath+"annotations/"+subset+"_Thi_"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["thi"] = np.mean(np.load(path), axis=0)

    return ANN


def load_network(kind, parcel, data="lau", hemi="both", binary=False,
                 version=1, subset="all", path=None):
    '''
    Function to load a network as well as its attributes
    ----------------------------------
    -> kind : Either SC or FC
    -> hemi : Either "both", "L" or "R"
    -> data : Either "HCP" or "lau"
    -> parcel : "68", "114", ... [if 'lau'] / "s400", "s800" [if "HCP"]
    -> version : either 1 (consensus computed without subcortex) or 2
        (consensus conputed with subcortex)
    -> subset : either 'discov', 'valid' or 'all'
    -> path : path to the "data" folder in which the data will be stored. If
        none, then assumes that path is current folder.
    '''

    mainPath = path+"/brainNetworks/"+data+"/"
    home = os.path.expanduser("~")

    Network = {}

    Network["info"] = {}
    Network["info"]["kind"] = kind
    Network["info"]["parcel"] = parcel
    Network["info"]["data"] = data
    Network["info"]["hemi"] = hemi
    Network["info"]["binary"] = binary
    Network["info"]["version"] = version
    Network["info"]["subset"] = subset

    if version == 1:
        version = ''
    else:
        version = "_v2"

    if binary is True:
        binary = "b"
    else:
        binary = ''

    if subset == "all":
        subset = ''

    mainPath = path+"/brainNetworks/"+data+"/"
    matrixPath = mainPath+"matrices/"+subset+kind+parcel+binary+version

    # hemisphere
    if hemi == "both":
        Network["hemi"] = np.load(matrixPath+"/hemi.npy")
        hemi = ''

    # Adjacency matrix
    # [set negative values to 0, fill diagonal, make symmetric]
    path = matrixPath+".npy"
    A = np.load(path)
    A[A < 0] = 0
    np.fill_diagonal(A, 0)
    A = (A + A.T)/2
    Network["adj"] = A

    # coordinates
    path = mainPath+"coords/coords"+parcel+hemi+".npy"
    Network["coords"] = np.load(path)

    # node strength
    Network["str"] = np.sum(A, axis=0)

    # Inverse of adjacency matrix
    inv = A.copy()
    inv[A > 0] = 1/inv[A > 0]
    Network["inv_adj"] = inv

    # distance
    Network["dist"] = cdist(Network["coords"], Network["coords"])

    # clustering coefficient
    Network["cc"] = bct.clustering_coef_wu(A)

    # shortest path
    path = matrixPath+"/sp.npy"
    if os.path.exists(path) is False:
        print("computing shortest paths... (might take a while)")
        Network["sp"] = bct.distance_wei(Network["adj"])[0]
        np.save(matrixPath+"/sp.npy", Network["sp"])
    else:
        Network["sp"] = np.load(path)

    # diffusion embedding
    de = compute_diffusion_map(A, n_components=10, return_result=True)
    Network["de"] = de[0]
    Network["de_extra"] = de[1]

    # eigenvector centrality
    Network["ec"] = bct.eigenvector_centrality_und(A)

    # mean first passage time
    Network["mfpt"] = bct.mean_first_passage_time(A)

    # betweenness centrality
    path = matrixPath+"/bc.npy"
    if os.path.exists(path) is False:
        print("computing betweenness centrality... (might take a while)")
        Network["bc"] = bct.betweenness_wei(Network["adj"])
        np.save(matrixPath+"/bc.npy", Network["bc"])
    else:
        Network["bc"] = np.load(path)

    # routing efficiency
    Network["r_eff"] = efficiency(Network)

    # diffusion efficiency
    Network["d_eff"] = efficiency_diffusion(Network)

    # subgraph centrality
    Network["subc"] = bct.subgraph_centrality(Network["adj"])

    # communities + participation coefficient
    path = matrixPath+"/communities/"
    if os.path.exists(path):
        files = []
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path, i)) and 'ci_' in i:
                files.append(i)
        if len(files) > 0:
            Network["ci"] = []
            for file in files:
                Network["ci"].append(np.load(os.path.join(path, file)))

            Network["ppc"] = []
            for i in range(len(files)):
                ppc = bct.participation_coef(A, Network["ci"][i])
                Network["ppc"].append(ppc)

    if (data == "HCP") and (kind == "SC"):
        path = mainPath+"matrices/"+subset+kind+parcel+hemi+"_lengths.npy"
        Network["lengths"] = np.load(path)

    # network information
    if parcel[0] == "s":
        n = parcel[1:]
        Network["order"] = "LR"
        Network["noplot"] = [b'Background+FreeSurfer_Defined_Medial_Wall',
                             b'']
        Network["lhannot"] = (home+"/"
                              "nnt-data/"
                              "atl-schaefer2018/"
                              "fsaverage/"
                              "atl-Schaefer2018_space-fsaverage_"
                              "hemi-L_desc-"+n+"Parcels7Networks_"
                              "deterministic.annot")
        Network["rhannot"] = (home+"/"
                              "nnt-data/"
                              "atl-schaefer2018/"
                              "fsaverage/"
                              "atl-Schaefer2018_space-fsaverage_"
                              "hemi-R_desc-"+n+"Parcels7Networks_"
                              "deterministic.annot")
    else:
        n = parcel_to_n(parcel)
        Network["order"] = "RL"
        Network["noplot"] = None
        Network["lhannot"] = (home+"/"
                              "nnt-data/"
                              "atl-cammoun2012/"
                              "fsaverage/"
                              "atl-Cammoun2012_space-fsaverage_"
                              "res-"+n+"_hemi-L_deterministic.annot")
        Network["rhannot"] = (home+"/"
                              "nnt-data/"
                              "atl-cammoun2012/"
                              "fsaverage/"
                              "atl-Cammoun2012_space-fsaverage_"
                              "res-"+n+"_hemi-R_deterministic.annot")

    return Network


def parcel_to_n(parcel):

    mapping = {}
    mapping["68"] = "033"
    mapping["114"] = "060"
    mapping["219"] = "125"
    mapping["448"] = "250"
    mapping["1000"] = "500"

    return mapping[parcel]


def efficiency(Network):

    sp = Network["sp"]
    n = len(sp)

    efficiency_local = np.zeros((n))
    for i in range(n):
        efficiency_local[i] = np.mean(1/np.delete(sp[i, :], i))
    efficiency_global = np.mean(efficiency_local)

    return efficiency_global


def efficiency_diffusion(Network):

    if isinstance(Network, dict):
        mfpt = Network["mfpt"]
    else:
        mfpt = bct.mean_first_passage_time(Network)
    n = len(mfpt)

    efficiency_local = np.zeros((n))
    for i in range(n):
        efficiency_local[i] = np.mean(1/np.delete(mfpt[i, :], i))
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
