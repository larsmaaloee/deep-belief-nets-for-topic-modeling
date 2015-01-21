__author__ = 'larsmaaloee'

from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib
from matplotlib import animation


def pca(X, class_indices):
    # Extract class names to python list,
    # then encode with integers (dict)
    classLabels = class_indices
    classNames = sorted(set(classLabels))
    classDict = dict(zip(classNames, range(len(classNames))))
    # Extract vector y, convert to NumPy matrix and transpose
    y = np.mat([classDict[value] for value in classLabels]).T
    # Compute values of N, M and C.
    N, _ = X.shape
    C = len(classNames)


    # Subtract mean value from data
    Y = X - np.ones((N, 1)) * X.mean(0)
    # PCA by computing SVD of Y
    _, _, V = linalg.svd(Y, full_matrices=False)
    V = mat(V).T
    Z = Y * V  # Project the centered data onto principal component space

    return C, y, Z


def pca_2d(X, class_indices, path, name, number_of_components, data_legend):
    C, y, Z = pca(X, class_indices)
    K = number_of_components  # Number of principal components to draw.

    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
    while True:
        if C <= len(markers):
            break
        markers += markers

    f = plt.figure(figsize=(10, 10), facecolor='white')
    f.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    N1 = ceil(sqrt(K))
    N2 = ceil(K / N1)

    k = 0
    for i in range(int(N1)):
        for j in xrange(i + 1, i + 1 + int(N2), 1):
            if i == j:
                continue
            k += 1
            ax = f.add_subplot(N1, N2, k, axisbg='white')

            frame = plt.gca()
            frame.axes.xaxis.set_ticklabels([])
            frame.axes.yaxis.set_ticklabels([])
            for c in range(C):
                # select indices belonging to class c:
                class_mask = y.A.ravel() == c
                ax.plot(Z[class_mask, i], Z[class_mask, j], linestyle='None', alpha=0.6, marker=markers[c],
                        markersize=6)

            plt.xlabel('PC{0}'.format(i + 1))
            plt.ylabel('PC{0}'.format(j + 1))

    legend = plt.legend(data_legend, loc=(-0.35, 0.8), labelspacing=1.1)
    plt.setp(legend.get_texts(), fontsize="small")

    savefig(path + '/' + name + '.png', dpi=220)
    show()


def pca_2d_for_2_components(X, component1, component2, class_indices, path, name, data_legend):
    font = {'family': 'sans-serif',
            'weight': 'light',
            'size': 22}
    matplotlib.rc('font', **font)

    C, y, Z = pca(X, class_indices)

    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
    while True:
        if C <= len(markers):
            break
        markers += markers

    f = plt.figure(figsize=(15, 15))
    f.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    ax = f.add_subplot(1, 1, 1, axisbg='white')

    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y.A.ravel() == c
        ax.plot(Z[class_mask, component1 - 1], Z[class_mask, component2 - 1], linestyle='None', alpha=1.0,
                marker=markers[c], markersize=6)

    # plt.xlabel('PC{0}'.format(component1))
    #plt.ylabel('PC{0}'.format(component2))

    legend = plt.legend(data_legend)
    #legend = plt.legend(data_legend,loc=(0.0,0.0),labelspacing = 0.1)

    savefig(path + '/' + name + '.png', dpi=200, transparent=True)
    show()


def pca_3d(X, component1, component2, component3, class_indices, path, name, data_legend):
    C, y, Z = pca(X, class_indices)

    colors = cm.rainbow(np.linspace(0, 1, C))
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
    while True:
        if C <= len(markers):
            break
        markers += markers

    # Plot PCA of the data
    f = plt.figure(figsize=(15, 15))
    ax = f.add_subplot(111, projection='3d', axisbg='white')
    ax._axis3don = False

    for c in range(C):
        # select indices belonging to class c:
        class_mask = y.A.ravel() == c
        xs = Z[class_mask, component1 - 1]
        xs = xs.reshape(len(xs)).tolist()[0]
        ys = Z[class_mask, component2 - 1]
        ys = ys.reshape(len(ys)).tolist()[0]
        zs = Z[class_mask, component3 - 1]
        zs = zs.reshape(len(zs)).tolist()[0]
        ax.scatter(xs, ys, zs, s=20, c=colors[c], marker=markers[c])


    # plt.figtext(0.5, 0.93, 'PCA 3D', ha='center', color='black', weight='light', size='large')



    f.savefig(path + '/' + name + '.png', dpi=200)
    plt.show()


def pca_3d_movie(X, component1, component2, component3, class_indices, path, name, data_legend):
    C, y, Z = pca(X, class_indices)

    colors = cm.rainbow(np.linspace(0, 1, C))
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
    while True:
        if C <= len(markers):
            break
        markers += markers

    # Plot PCA of the data
    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111, projection='3d', axisbg="black")
    ax._axis3don = False

    plt.figtext(0.5, 0.93, '', ha='center', color='black', weight='light', size='large')

    def init():
        for c in range(C):
            # select indices belonging to class c:
            class_mask = y.A.ravel() == c
            xs = Z[class_mask, component1 - 1]
            xs = xs.reshape(len(xs)).tolist()[0]
            ys = Z[class_mask, component2 - 1]
            ys = ys.reshape(len(ys)).tolist()[0]
            zs = Z[class_mask, component3 - 1]
            zs = zs.reshape(len(zs)).tolist()[0]
            ax.scatter(xs, ys, zs, s=20, c=colors[c], marker=markers[c])

    def animate(i):
        ax.view_init(elev=10., azim=i)

    anim = animation.FuncAnimation(f, animate, init_func=init,
                                   frames=360, interval=40, blit=True)
    # Save
    anim.save('output/animation.mp4', writer="ffmpeg", dpi=200)
    plt.show()