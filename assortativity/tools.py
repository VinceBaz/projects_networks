
import numpy as np
import matplotlib.pyplot as plt
from . import measures as m
import random

def plot_gradient(partition, coords, label=None, min_color=None, max_color=None,
                  colormap="viridis", colorbar=True,size=220, show=True, save=False, dpi=100, mac=False, norm=None):

    if min_color==None:
        min_color=np.amin(partition)
    if max_color==None:
        max_color = np.amax(partition)

    if mac==False:
        fig1 = plt.figure(figsize=(22,12),dpi=dpi)
    else:
        fig1 = plt.figure(dpi=dpi)
    ax1 = plt.subplot(121, projection='3d')
    mapp1 = ax1.scatter(xs=coords[:,0],
                       ys=coords[:,1],
                       zs=coords[:,2],
                       c = partition,
                       vmin = min_color,
                       vmax = max_color,
                       s=size,
                       cmap=colormap,
                       edgecolors='k',
                       norm=norm
                       )
    if colorbar==True:
        cbar = fig1.colorbar(mapp1, orientation = 'horizontal', pad=-0.05 )
        if label is not None:
            cbar.set_label(label)
    ax1.grid(True)
    ax1.axis('off')
    if mac==False:
        ax1.view_init(0,270)
        ax1.set(xlim=0.59 * np.array(ax1.get_xlim()),
            ylim=0.59 * np.array(ax1.get_ylim()),
            zlim=0.60 * np.array(ax1.get_zlim()),
            aspect=0.55)
        ax1.set(aspect=0.55) #0.8
    else:
        ax1.view_init(0,0)
    # plt.colorbar(mapp1, aspect=20) #shrink=0.5,
    plt.gca().patch.set_facecolor('white')
    if show is False:
        plt.close()
    if save is True:
        save_figure()
    fig2 = plt.figure(figsize=(22,12),dpi=dpi)
    ax1 = plt.subplot(121, projection='3d')
    mapp1 = ax1.scatter(xs=coords[:,0],
                       ys=coords[:,1],
                       zs=coords[:,2],
                       vmin = min_color,
                       vmax = max_color,
                       c = partition,
                       s=size,
                       cmap=colormap,
                       edgecolors='k',
                       norm=norm
                       )
    ax1.view_init(0,180)
    ax1.grid(True)
    ax1.axis('off')
    ax1.set(xlim=0.59 * np.array(ax1.get_xlim()),
           ylim=0.59 * np.array(ax1.get_ylim()),
           zlim=0.60 * np.array(ax1.get_zlim()),
           aspect=0.55)
    ax1.set(aspect=0.55) #0.8
    # plt.colorbar(mapp1, aspect=20) #shrink=0.5,
    plt.gca().patch.set_facecolor('white')

    if show is False:
        plt.close()
    if save is True:
        save_figure()
    fig3 = plt.figure(figsize=(22,12),dpi=dpi)
    ax1 = plt.subplot(121, projection='3d')
    mapp1 = ax1.scatter(xs=coords[:,0],
                       ys=coords[:,1],
                       zs=coords[:,2],
                       vmin = min_color,
                       vmax = max_color,
                       c = partition,
                       s=size,
                       cmap=colormap,
                       edgecolors='k',
                       norm=norm
                       )
    ax1.view_init(90,180)
    ax1.grid(True)
    ax1.axis('off')
    ax1.set(xlim=0.59 * np.array(ax1.get_xlim()),
           ylim=0.59 * np.array(ax1.get_ylim()),
           zlim=0.60 * np.array(ax1.get_zlim()),
           aspect=0.55)
    ax1.set(aspect=0.55) #0.8
    # plt.colorbar(mapp1, aspect=20) #shrink=0.5,
    plt.gca().patch.set_facecolor('white')

    if show is False:
        plt.close()
    if save is True:
        save_figure()

def assort_preserv_swap(A, M, g):

    nbEdges = np.sum(A)/2

    assort = m.globalAssort(A, M)

    #Remember previous graph in the sequence of graphs
    prevG = A

    #Stores if swap was successful, and the edges that are swapped
    swapped = []
    swapped.append(True)
    edgeSwapped = []

    #while assort is less than goal assort
    while assort<g:

        edges = np.where(prevG==1)

        #Choose 2 edges at random
        rand1 = random.randrange(0,len(edges[0]))
        rand2 = random.randrange(0,len(edges[0]))
        e1 = [edges[0][rand1], edges[1][rand1]]
        e2 = [edges[0][rand2], edges[1][rand2]]

        #if swap creates a self-loop or a multiedge, resample
        if e1[0] == e2[1] or e2[0] == e1[1]:
            swapped.append(False)
        elif prevG[e1[0], e2[1]]==1 or prevG[e2[0], e1[1]]==1:
            swapped.append(False)
        else:
            #swap edges
            newG = prevG.copy()
            newG[e1[0], e1[1]] = 0
            newG[e1[1], e1[0]] = 0
            newG[e2[0], e2[1]] = 0
            newG[e2[1], e2[0]] = 0
            newG[e1[0], e2[1]] = 1
            newG[e2[1], e1[0]] = 1
            newG[e2[0], e1[1]] = 1
            newG[e1[1], e2[0]] = 1

            #If new graph increases assortativity, keep swap
            assortNew = m.globalAssort(newG, M)
            if assortNew-assort>0:
                prevG = newG
                swapped.append(True)
                edgeSwapped.append([e1, e2])
                assort = assortNew
            else:
                swapped.append(False)

    return prevG, swapped, edgeSwapped
