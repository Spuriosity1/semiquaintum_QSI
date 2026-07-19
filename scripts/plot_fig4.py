from plot_heat_capacity import plot_file
from plot_hhl import *

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

search=r"../out/sq_qsi/run_Tc0.01_CQBRW_sw1024_samp128_L8_p0.050_Jzz1.000_Jxx*_merge*.davg.h5"

# labels=[r'$J_\pm/J_{zz}=-0.050$',r'$J_\pm/J_{zz}=0.000$',r'$J_\pm/J_{zz}=0.050$',r'$J_\pm/J_{zz}=0.100$',r'$J_\pm/J_{zz}=0.200$']
labels=[r'$j_\pm=-0.05$',r'$j_{\pm}=0.00$',r'$j_{\pm}=0.05$',r'$j_{\pm}=0.10$',r'$j_{\pm}=0.20$']

fnames=[
    r"../out/sq_qsi/run_Tc0.01_CQBRW_sw1024_samp128_L8_p0.050_Jzz1.000_Jxx-0.050_Jyy-0.050_merge8.davg.h5",
    r"../out/sq_qsi/run_Tc0.01_CQBRW_sw1024_samp128_L8_p0.050_Jzz1.000_Jxx0.000_Jyy0.000_merge8.davg.h5",
    r"../out/sq_qsi/run_Tc0.01_CQBRW_sw1024_samp128_L8_p0.050_Jzz1.000_Jxx0.050_Jyy0.050_merge8.davg.h5",
    r"../out/sq_qsi/run_Tc0.01_CQBRW_sw1024_samp128_L8_p0.050_Jzz1.000_Jxx0.100_Jyy0.100_merge8.davg.h5", 
    r"../out/sq_qsi/run_Tc0.01_CQBRW_sw1024_samp128_L8_p0.050_Jzz1.000_Jxx0.200_Jyy0.200_merge8.davg.h5"
    ]

fname_Szz="../out/sq_qsi/run_Tc0.01_CQBRW_sw1024_samp128_L8_p0.050_Jzz1.000_Jxx0.200_Jyy0.200_ds903_merge128.mavg.h5"
fname_Spm_pos="../out/sq_qsi/run_Tc0.01_CQBRW_sw1024_samp128_L8_p0.050_Jzz1.000_Jxx0.200_Jyy0.200_ds903_merge128.mavg.h5"
fname_Spm_neg="../out/sq_qsi/run_Tc0.01_CQBRW_sw1024_samp128_L8_p0.050_Jzz1.000_Jxx-0.050_Jyy-0.050_ds903_merge128.mavg.h5"

fig, Ax = plt.subplots(2,2, figsize=(3.4,3.2))


n_total = len(fnames) 
cmap    = plt.colormaps["tab10"]
colors  = [cmap(i) for i in range(n_total)]

ax_C = Ax[0,0]
ax_C.set_ylabel("Specific heat $C/N$")
ax_C.set_xlabel("Temperature $T/J_{zz}$")
ax_C.set_xscale("log")
ax_C.set_yscale("log")
ax_C.grid(True)
# ax_C.figure.tight_layout()


for fname, color, label in zip(fnames, colors, labels):
    plot_file(fname, {'C': ax_C}, color, label, None)


ax_C.legend(fontsize=6)


use_logscale = False
clim_szz=[0, 2]
clim_spm=[0, 2]


# <SzSz>
T_list, n_samples, corr, sl_pos, n_spins, k_dims = read_ssf(fname_Szz)
B = make_B(k_dims)
q_vecs, h_vals, l_vals, grid_shape = make_hhl_qvecs(k_dims)

data_flat = contract_at_qlist(corr, k_dims, B, sl_pos, q_vecs)

n_T = data_flat.shape[0]
T_list = T_list[:n_T]                                          # trim to match
S_hhl = data_flat.reshape(n_T, *grid_shape) / n_spins  # (n_T, n_h, n_l)



t_idx=-1
mesh = plot_panel(Ax[0,1], S_hhl[t_idx], h_vals, l_vals,
           T_list[t_idx], 'Szz', use_logscale, clim_szz)



# S+ S-


T_list, n_samples, corr_tcm, disp_vectors, n_pairs, n_quantum_spins = \
    read_tcm(fname_Spm_pos)
with h5py.File(fname_Spm_pos, "r") as f:
    k_dims = tuple(int(d) for d in f["/ssf"].attrs["k_dims"])

B = make_B(k_dims)
q_vecs, h_vals, l_vals, grid_shape = make_hhl_qvecs(k_dims)
data_flat = compute_spm_at_qlist(
    corr_tcm, n_samples, disp_vectors, n_quantum_spins, q_vecs
)

n_T = data_flat.shape[0]
T_list = T_list[:n_T]                                          # trim to match
S_hhl = data_flat.reshape(n_T, *grid_shape)  # (n_T, n_h, n_l)

t_idx=-1
plot_panel(Ax[1,1], S_hhl[t_idx], h_vals, l_vals,
           T_list[t_idx], 'Spm', use_logscale, clim_spm)




T_list, n_samples, corr_tcm, disp_vectors, n_pairs, n_quantum_spins = \
    read_tcm(fname_Spm_neg)
with h5py.File(fname_Spm_neg, "r") as f:
    k_dims = tuple(int(d) for d in f["/ssf"].attrs["k_dims"])

B = make_B(k_dims)
q_vecs, h_vals, l_vals, grid_shape = make_hhl_qvecs(k_dims)
data_flat = compute_spm_at_qlist(
    corr_tcm, n_samples, disp_vectors, n_quantum_spins, q_vecs
)

n_T = data_flat.shape[0]
T_list = T_list[:n_T]                                          # trim to match
S_hhl = data_flat.reshape(n_T, *grid_shape)  # (n_T, n_h, n_l)

t_idx=-1
plot_panel(Ax[1,0], S_hhl[t_idx], h_vals, l_vals,
           T_list[t_idx], 'Spm', use_logscale, clim_spm)

padx = 0.05
pady = 0.05

for a in Ax.ravel():
    a.set_title('')

a = Ax[0,0]
a.text(padx, 1-pady, '(a)', color='k', ha='left', va='top', transform=a.transAxes)
a = Ax[0,1]
a.text(padx, 1-pady, r'(b)',
       color='w', ha='left', va='top', transform=a.transAxes)
a.text(padx, pady, r'$\mathcal{S}^{zz}~j_\pm=0.20$', fontsize=8,
       color='w', ha='left', va='bottom', transform=a.transAxes)
a = Ax[1,1]
a.text(padx, 1-pady, r'(d)',
       color='w', ha='left', va='top', transform=a.transAxes)
a.text(padx, pady, r'$\mathcal{S}^{+-}~j_\pm=0.20$', fontsize=8,
       color='w', ha='left', va='bottom', transform=a.transAxes)
a = Ax[1,0]
a.text(padx, 1-pady, r'(c)',
       color='w', ha='left', va='top', transform=a.transAxes)
a.text(padx, pady, r'$\mathcal{S}^{+-}~j_\pm=-0.05$', fontsize=8,
       color='w', ha='left', va='bottom', transform=a.transAxes)

fig.tight_layout()
fig.subplots_adjust(right=0.99, top=0.97, wspace=0.3, hspace=0.444, bottom=0.127)

#cax=fig.add_axes([0.6,0.5,0.3,0.03])
#cb = plt.colorbar(mesh, cax=cax, orientation="horizontal")

cax=fig.add_axes([0.08,0.15,0.03,0.26])
cb = plt.colorbar(mesh, cax=cax, orientation="vertical")
cax.tick_params(labelsize=8, pad=1)
cax.set_yticks(np.arange(0,2.1, 0.1), minor=True)
cax.yaxis.set_ticks_position("left")

fig.savefig("/Users/alaricsanders/Desktop/defect_figs/Fig4_raw.pdf")

plt.show()
