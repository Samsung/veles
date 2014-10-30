import os
import sys
import matplotlib
matplotlib.use('cairo')
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy
from scipy.cluster.hierarchy import linkage, leaves_list
import xmltodict


if __name__ == "__main__":
    with open(sys.argv[1], 'r', encoding="utf-8") as fin:
        data = xmltodict.parse(fin.read())

    files = list(sorted(set.union({dup['file'][i]['@path']
                                   for dup in data['pmd-cpd']['duplication']
                                   for i in (0, 1)})))
    findex = {f: i for i, f in enumerate(files)}
    mat = numpy.zeros((len(files), len(files)))
    for dup in data['pmd-cpd']['duplication']:
        mat[tuple(findex[dup['file'][i]['@path']] for i in (0, 1))] += \
            int(dup['@lines'])
    mat += mat.transpose()

    mat[mat == 0] = 0.001  # any value << 1
    # cluster the distances matrix and get the expressive indices order
    order = leaves_list(linkage(1 / mat))
    # apply the new order
    mat = mat[numpy.ix_(order, order)]
    files = [files[i] for i in order]

    # construct the linear gradient map white -> red
    cdict = {'red':   ((0.0, 1.0, 1.0),
                       (1.0, 1.0, 1.0)),
             'green': ((0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0)),
             'blue':  ((0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0))}
    reds = LinearSegmentedColormap('Reds', cdict)

    # draw the map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(mat, cmap=reds)
    # uncomment the following to remove the frame around the map
    # ax.set_frame_on(False)
    ax.set_xlim((0, len(files)))
    ax.set_ylim((0, len(files)))
    ax.set_xticks(numpy.arange(len(files)) + 0.5, minor=False)
    ax.set_yticks(numpy.arange(len(files)) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels([os.path.basename(f) for f in files], minor=False,
                       rotation=90)
    ax.set_yticklabels([os.path.basename(f) for f in files], minor=False)
    ax.grid(False)
    ax.set_aspect(1)
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    fig_size = 16 * len(files) / 55
    fig.set_size_inches(fig_size, fig_size)
    plt.savefig(sys.argv[2], bbox_inches='tight', transparent=False, dpi=100,
                pad_inches=0.1)
