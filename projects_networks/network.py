# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:34:05 2020

Class for the network representation of brain

@author: Vincent Bazinet
"""

import os
import bct
import numpy as np
from scipy.spatial.distance import cdist
from mapalign.embed import compute_diffusion_map
from . import load_data


class Network(object):
    '''
    Class for the network representation of brain

    Parameters
    ----------
    kind : string
        Either 'SC' or 'FC'.
    hemi : string
        Either "both", "L" or "R".
    data : string
        Either "HCP" or "lau".
    parcel : string
        Either "68", "114", ... [if 'lau'] / "s400", "s800" [if "HCP"]
    version : int
        Either 1 (consensus computed without subcortex) or 2 (consensus
        conputed with subcortex)
    subset : string
        Either 'discov', 'valid' or 'all'
    path : string
        path to the "data" folder in which the data will be stored. If
        none, then assumes that path is current folder.

    Attributes
    ----------
    info : dict
        Dictionary of the parameters used to construct the network.
    hemi :
    adj :

    '''
    def __init__(self, kind, parcel, data='lau', hemi='both', binary=False,
                 version=1, subset='all', path=None):

        mainPath = path+"/brainNetworks/"+data+"/"
        home = os.path.expanduser("~")

        self.info = {}
        self.info["kind"] = kind
        self.info["parcel"] = parcel
        self.info["data"] = data
        self.info["hemi"] = hemi
        self.info["binary"] = binary
        self.info["version"] = version
        self.info["subset"] = subset

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

        if hemi == "both":
            hemi = ''

        matrxPath = mainPath+"matrices/"+subset+kind+parcel+hemi+binary+version

        # hemisphere
        self.hemi = np.load(matrxPath+"/hemi.npy")

        # Adjacency matrix
        path = matrxPath+".npy"
        A = np.load(path)

        # Look at time when file was last modified
        last_modified = os.path.getmtime(path)

        # set negative values to 0, fill diagonal, make symmetric
        A[A < 0] = 0
        np.fill_diagonal(A, 0)
        A = (A + A.T)/2
        self.adj = A

        # Number of nodes in the network
        self.n = len(self.adj)

        # coordinates
        path = mainPath+"coords/coords"+parcel+hemi+".npy"
        self.coords = np.load(path)

        # Inverse of adjacency matrix
        inv = A.copy()
        inv[A > 0] = 1/inv[A > 0]
        self.inv_adj = inv

        # distance
        self.dist = cdist(self.coords, self.coords)

        # shortest path
        #
        # Loaded from saved file...
        # IF file not found OR Adjacency was modified after creation,
        # then recompute measure
        path = matrxPath+"/sp.npy"

        if os.path.exists(path) is False:
            print("shortest path not found")
            print("computing shortest path...")
            self.sp = bct.distance_wei(self.inv_adj)[0]
            np.save(matrxPath+"/sp.npy", self.sp)

        elif os.path.getmtime(path) < last_modified:
            print("new adjacency matrix was found")
            print("computing shortest paths...")
            self.sp = bct.distance_wei(self.inv_adj)[0]
            np.save(matrxPath+"/sp.npy", self.sp)

        else:
            self.sp = np.load(path)

        # diffusion embedding
        de = compute_diffusion_map(A, n_components=10, return_result=True)
        self.de = de[0]
        self.de_extra = de[1]

        # Principal components
        self.PCs, self.PCs_ev = load_data.getPCs(self.adj)

        # betweenness centrality
        #
        # Loaded from saved file...
        # IF file not found OR Adjacency was modified after creation,
        # then recompute measure
        path = matrxPath+"/bc.npy"
        if os.path.exists(path) is False:

            print("betweenness centrality not found")
            print("computing betweenness centrality...")
            self.bc = bct.betweenness_wei(self.inv_adj)
            np.save(matrxPath+"/bc.npy", self.bc)

        elif os.path.getmtime(path) < last_modified:
            print("new adjacency matrix was found")
            print("recomputing betweeness centrality...")
            self.bc = bct.betweenness_wei(self.inv_adj)
            np.save(matrxPath+"/bc.npy", self.bc)

        else:
            self.bc = np.load(path)

        # communities + participation coefficient
        path = matrxPath+"/communities/"
        if os.path.exists(path):
            files = []
            for i in os.listdir(path):
                if os.path.isfile(os.path.join(path, i)) and 'ci_' in i:
                    files.append(i)
            if len(files) > 0:
                self.ci = []
                for file in files:
                    self.ci.append(np.load(os.path.join(path, file)))

                self.ppc = []
                for i in range(len(files)):
                    ppc = bct.participation_coef(A, self.ci[i])
                    self.ppc.append(ppc)

        if (data == "HCP") and (kind == "SC"):
            path = mainPath+"matrices/"+subset+kind+parcel+hemi+"_lengths.npy"
            self.lengths = np.load(path)

        # streamline connection lengths
        path = matrxPath+"/len.npy"
        if os.path.exists(path) is True:
            self.len = np.load(path)

        # network information
        if parcel[0] == "s":
            nb = parcel[1:]
            self.order = "LR"
            self.noplot = [b'Background+FreeSurfer_Defined_Medial_Wall',
                           b'']
            self.lhannot = (home+"/"
                            "nnt-data/"
                            "atl-schaefer2018/"
                            "fsaverage/"
                            "atl-Schaefer2018_space-fsaverage_"
                            "hemi-L_desc-"+nb+"Parcels7Networks_"
                            "deterministic.annot")
            self.rhannot = (home+"/"
                            "nnt-data/"
                            "atl-schaefer2018/"
                            "fsaverage/"
                            "atl-Schaefer2018_space-fsaverage_"
                            "hemi-R_desc-"+nb+"Parcels7Networks_"
                            "deterministic.annot")
        else:
            nb = _parcel_to_n(parcel)
            self.order = "RL"
            self.noplot = None
            self.lhannot = (home+"/"
                            "nnt-data/"
                            "atl-cammoun2012/"
                            "fsaverage/"
                            "atl-Cammoun2012_space-fsaverage_"
                            "res-"+nb+"_hemi-L_deterministic.annot")
            self.rhannot = (home+"/"
                            "nnt-data/"
                            "atl-cammoun2012/"
                            "fsaverage/"
                            "atl-Cammoun2012_space-fsaverage_"
                            "res-"+nb+"_hemi-R_deterministic.annot")
            self.cammoun_id = nb

    @property
    def strength(self):
        """Node strength"""
        return np.sum(self.adj, axis=0)

    @property
    def cc(self):
        """Clustering coefficient"""
        return bct.clustering_coef_wu(self.adj)

    @property
    def clo(self):
        """Closeness centrality"""
        return 1/np.mean(self.sp, axis=0)

    @property
    def subc(self):
        """subgraph centrality"""
        return bct.subgraph_centrality(self.adj)

    @property
    def mfpt(self):
        """mean first passage time"""
        return bct.mean_first_passage_time(self.adj)

    @property
    def ec(self):
        """eigenvector centrality"""
        return bct.eigenvector_centrality_und(self.adj)

    @property
    def r_eff(self):
        """routing efficiency"""
        r_eff_loc = np.zeros((self.n))
        for i in range(self.n):
            r_eff_loc[i] = np.mean(1/np.delete(self.sp[i, :], i))
        r_eff_glo = np.mean(r_eff_loc)

        return r_eff_glo

    @property
    def d_eff(self):
        """diffusion efficiency"""
        mfpt = self.mfpt()
        d_eff_loc = np.zeros((self.n))
        for i in range(self.n):
            d_eff_loc[i] = np.mean(1/np.delete(mfpt[i, :], i))
        d_eff_glo = np.mean(d_eff_loc)

        return d_eff_glo


def _parcel_to_n(parcel):
    mapping = {}
    mapping["68"] = "033"
    mapping["114"] = "060"
    mapping["219"] = "125"
    mapping["448"] = "250"
    mapping["1000"] = "500"
    return mapping[parcel]
