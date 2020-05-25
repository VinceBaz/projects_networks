import random
import numpy as np
from . import assortativity as m


def assort_preserv_swap(A, M, g, weighted=False, verbose=0):

    if weighted is False:
        assortativity = m.globalAssort
    else:
        assortativity = m.weightedAssort

    assort = assortativity(A, M)

    # Remember previous graph in the sequence of graphs
    prevG = A

    # Stores if swap was successful, and the edges that are swapped
    swapped = []
    swapped.append(True)
    edgeSwapped = []

    # while assort is less than goal assort
    while assort < g:

        edges = np.where(prevG > 0)

        # Choose 2 edges at random
        rand1 = random.randrange(0, len(edges[0]))
        rand2 = random.randrange(0, len(edges[0]))
        e1 = [edges[0][rand1], edges[1][rand1]]
        e2 = [edges[0][rand2], edges[1][rand2]]

        # if swap creates a self-loop or a multiedge, resample
        if e1[0] == e2[1] or e2[0] == e1[1]:
            if verbose > 1:
                print("self-loop...")
            swapped.append(False)
        elif prevG[e1[0], e2[1]] > 0 or prevG[e2[0], e1[1]] > 0:
            if verbose > 1:
                print("multi-edge...")
            swapped.append(False)
        else:
            # swap edges
            newG = prevG.copy()
            newG[e1[0], e1[1]] = 0
            newG[e1[1], e1[0]] = 0
            newG[e2[0], e2[1]] = 0
            newG[e2[1], e2[0]] = 0
            newG[e1[0], e2[1]] = prevG[e1[0], e1[1]]
            newG[e2[1], e1[0]] = prevG[e1[0], e1[1]]
            newG[e2[0], e1[1]] = prevG[e2[0], e2[1]]
            newG[e1[1], e2[0]] = prevG[e2[0], e2[1]]

            # If new graph increases assortativity, keep swap
            assortNew = assortativity(newG, M)

            if verbose > 1:
                print(assortNew)

            if assortNew - assort > 0:
                prevG = newG
                swapped.append(True)
                edgeSwapped.append([e1, e2])
                assort = assortNew

                if verbose > 0:
                    print(assort)

            else:
                swapped.append(False)

        if verbose > 1:
            print(swapped[-1])

    return prevG, swapped, edgeSwapped
