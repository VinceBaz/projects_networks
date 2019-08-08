
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

