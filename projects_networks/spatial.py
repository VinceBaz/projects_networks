import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zscore

def localI(A, coords, M):
    '''
    Function to compute the local Moran's I of the nodes
    in a network
    A -> Adjacency Matrix
    coords -> Coordinates of the nodes in the network
    N -> Vector of attributes
    '''

    n = len(M)  # nb of nodes in network

    #get weights matrix (inverse of eucledian distance)
    d = cdist(coords, coords)
    d = 1/d
    np.fill_diagonal(d, 0)
    
    #row normalize the weights
    rsd = d/ (d.sum(axis=1)[:,np.newaxis])
    
    #get difference of attributes
    z = M - np.mean(M)
    
    il = np.zeros((n))
    for i in range(n):
        num = 0
        for j in range(n):
            num = num + (z[i] * rsd[i,j] * z[j])
            
        den = 0
        for j in range(n):
            den = den + (z[j] * z[j])
        
        il[i] = (num/den) * n
        
    ig = np.mean(il)
    
    return il, ig