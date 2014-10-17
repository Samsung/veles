import os
import sys
import matplotlib
matplotlib.use('cairo')
from matplotlib import pyplot as plt, cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy
import xmltodict


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as fin:
        data = xmltodict.parse(fin.read())

    files = set((dup['file'][0]['@path']
                 for dup in data['pmd-cpd']['duplication']))
    files = files.union(set((dup['file'][1]['@path']
                             for dup in data['pmd-cpd']['duplication'])))
    files = list(sorted(files))
    findex = {f: i for i, f in enumerate(files)}
    mat = numpy.zeros((len(files), len(files)))
    for dup in data['pmd-cpd']['duplication']:
        lines = dup['@lines']
        i1 = findex[dup['file'][0]['@path']]
        i2 = findex[dup['file'][1]['@path']]
        mat[i1, i2] += int(lines)
        mat[i2, i1] += int(lines)

    cdict = {'red':   ((0.0, 1.0, 1.0),
                       (1.0, 1.0, 1.0)),
             'green': ((0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0)),
             'blue':  ((0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0))}
    reds = LinearSegmentedColormap('Reds', cdict)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(mat, cmap=reds)
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

    fig.set_size_inches(16, 16)
    plt.savefig(sys.argv[2], bbox_inches='tight', transparent=False, dpi=100,
                pad_inches=0)
