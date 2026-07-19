from plot_percol_transitions import plot_transitions
from plot_hist import plot_histfile
import matplotlib.pyplot as plt
import glob

tex_fonts_ss = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{sfmath}\renewcommand{\familydefault}{\sfdefault}",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}


tex_fonts_serif = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}


plt.rcParams.update(tex_fonts_serif)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(3.4,3), height_ratios=[1.4,1])


ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r"Cluster size $s$")
ax1.set_ylabel(r"$s\cdot n(s)$")

histfiles=glob.glob("../../out/percol_CMC/hist_L20*_merge128*_nn24.h5")


plot_histfile(ax1, histfiles, True, True) 

plot_transitions(ax2)

fig.tight_layout()
fig.subplots_adjust(top=0.97,bottom=0.132, left=0.175, right=0.95, hspace=0.4)

ax1.legend(prop={'size': 8})

fig.savefig("/Users/alaricsanders/Desktop/defect_figs/Fig3_raw.pdf")
plt.show()

