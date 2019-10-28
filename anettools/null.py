import random
import numpy as np
from . import assortativity as m

def assort_preserv_swap(A, M, g, verbose=False):

    assort = m.globalAssort(A, M)

    # Remember previous graph in the sequence of graphs
    prevG = A

    # Stores if swap was successful, and the edges that are swapped
    swapped = []
    swapped.append(True)
    edgeSwapped = []

    # while assort is less than goal assort
    while assort < g:

        edges = np.where(prevG == 1)

        # Choose 2 edges at random
        rand1 = random.randrange(0, len(edges[0]))
        rand2 = random.randrange(0, len(edges[0]))
        e1 = [edges[0][rand1], edges[1][rand1]]
        e2 = [edges[0][rand2], edges[1][rand2]]

        # if swap creates a self-loop or a multiedge, resample
        if e1[0] == e2[1] or e2[0] == e1[1]:
            swapped.append(False)
        elif prevG[e1[0], e2[1]] == 1 or prevG[e2[0], e1[1]] == 1:
            swapped.append(False)
        else:
            # swap edges
            newG = prevG.copy()
            newG[e1[0], e1[1]] = 0
            newG[e1[1], e1[0]] = 0
            newG[e2[0], e2[1]] = 0
            newG[e2[1], e2[0]] = 0
            newG[e1[0], e2[1]] = 1
            newG[e2[1], e1[0]] = 1
            newG[e2[0], e1[1]] = 1
            newG[e1[1], e2[0]] = 1

            # If new graph increases assortativity, keep swap
            assortNew = m.globalAssort(newG, M)
            if assortNew - assort > 0:
                prevG = newG
                swapped.append(True)
                edgeSwapped.append([e1, e2])
                assort = assortNew

                if verbose is True:
                    print(assort)

            else:
                swapped.append(False)

    return prevG, swapped, edgeSwapped
