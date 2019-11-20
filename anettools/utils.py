# -*- coding: utf-8 -*-
"""
Created on : 2019/11/20
Last updated on: 
@author: Vincent Bazinet
"""

import numpy as np

def isBipartite(A):
    
    n = len(A)
    
    # Create a color array to store colors  
    # assigned to all veritces. Vertex 
    # number is used as index in this array.  
    # The value '-1' of  colorArr[i] is used to  
    # indicate that no color is assigned to  
    # vertex 'i'. The value 1 is used to indicate  
    # first color is assigned and value 0 
    # indicates second color is assigned. 
    colorArr = np.zeros((n))-1
  
    # Assign first color to source 
    colorArr[0] = 1
  
    # Create a queue (FIFO) of vertex numbers and  
    # enqueue source vertex for BFS traversal 
    queue = [] 
    queue.append(0) 
  
    # Run while there are vertices in queue  
    # (Similar to BFS) 
    while queue: 
  
        u = queue.pop() 
  
        # Return false if there is a self-loop 
        if A[u,u] == 1: 
            return False; 
  
        for v in range(n): 
  
            # An edge from u to v exists and destination  
            # v is not colored 
            if A[u,v] == 1 and colorArr[v] == -1: 
  
                # Assign alternate color to this  
                # adjacent v of u 
                colorArr[v] = 1 - colorArr[u] 
                queue.append(v) 
  
            # An edge from u to v exists and destination  
            # v is colored with same color as u 
            elif A[u,v] == 1 and colorArr[v] == colorArr[u]: 
                return False
  
    # If we reach here, then all adjacent  
    # vertices can be colored with alternate  
    # color 
    return True