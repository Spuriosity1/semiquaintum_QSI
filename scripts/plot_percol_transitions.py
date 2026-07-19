import sqlite3
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.ticker as mtick


tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "sans-serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}


def plot_transitions(ax):

    transitions = {
        'QSI links': (0.132, 0.002),
        'QSI plaq.': (0.100, 0.003),
        'QSI vol.': (0.060, 0.004)
    }

    textcolor='w'
    barheight=1


    ys=[0,-1,-2]

    xlim=0.25

    for y, k in zip(ys, transitions):
        pc, delta_pc = transitions[k]
        ax.barh(y, width=pc,xerr=delta_pc, height=barheight)
        ax.text(0.005, y, s=k,va='center', color=textcolor)

    clust_percol = {
            r'$O(j_\pm)$ clusters': (0.100, 0.002),
            r'$O(j_\pm^2)$ clusters': (0.018, 0.001)
            }


    for y, k in zip([-3,-4], clust_percol):
        w, werr = clust_percol[k]
        ax.barh(y, width=1-w, left=w,height=barheight)
        ax.barh(y, width=w, color=(0,0,0,0), xerr=werr, height=barheight)
        ax.text(xlim-0.01, y, s=k, va='center', ha='right', color=textcolor)



    ax.set_xlabel('Disorder $p$')

    #ax.set_yticks([0,-1,-2,-3,-4], labels=list(transitions.keys()) + ['2nn clust.','4nn clust.'])
    ax.set_yticks([])

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0,decimals=0))
    ax.set_xticks(np.arange(0,0.26, 0.01),minor=True)

    ax.tick_params(which='both', top=True, labeltop=False, bottom=True, labelbottom=True)


    ax.set_ylim([-4.5, 0.5])

    ax.set_xlim([0,xlim])

if __name__=="__main__":

    plt.rcParams.update(tex_fonts)

    fig,ax=plt.subplots(figsize=(3, 1.8))

    plot_transitions(ax)

    fig.tight_layout()

    fig.savefig("/Users/alaricsanders/Desktop/defect_figs/percol_transitions.pdf")

    plt.show()
