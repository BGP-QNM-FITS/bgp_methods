import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from CCE import SXS_CCE
import bgp_qnm_fits as bgp

from plot_config import PlotConfig
config = PlotConfig()
config.apply_style()

colors = config.colors
custom_colormap = LinearSegmentedColormap.from_list("custom_colormap", colors)

ID = "0001"

T0s = np.linspace(0, 70, 30)
T = 100 
N_MAX = 3 
sim_main = SXS_CCE(ID, lev="Lev5", radius="R2")
Mf_ref = sim_main.Mf
chif_mag_ref = sim_main.chif_mag

qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)] + [(3,2,0,1)]
spherical_modes = [(2, 2)]
tuned_param_dict_GP = bgp.get_param_data("GP")[ID]

fits_GP = bgp.BGP_fit(
    sim_main.times,
    sim_main.h,
    qnm_list,
    Mf_ref,
    chif_mag_ref,
    tuned_param_dict_GP,
    bgp.kernel_GP,
    t0=T0s,
    use_nonlinear_params=False,
    decay_corrected=True,
    num_samples=100000,
    t0_method="closest",
    T=T,
    spherical_modes=spherical_modes,
    include_chif=False,
    include_Mf=False,
)

qnm_indices = [0, -1, 1, 2, 3] 

fig, axes = plt.subplots(5, 1, figsize=(config.fig_width, config.fig_height * 5))
colors = custom_colormap(np.linspace(0, 1, len(qnm_list)))

for j, qnm_index in enumerate(qnm_indices):

    sampless = [] 

    for i, fit in enumerate(fits_GP.fits):
        samples = fit["sample_amplitudes"][:,qnm_index].tolist()
        sampless.append(samples)

    axes[j].violinplot(sampless, positions=T0s, widths=2, showmedians=True)

axes[0].set_ylim(0.97,1)
axes[2].set_ylim(3,6)
axes[3].set_ylim(0,30)
axes[4].set_ylim(0,40)

axes[0].set_ylabel(r"$\mathcal{A}_{2,2,0,+}$")
axes[1].set_ylabel(r"$\mathcal{A}_{3,2,0,+}$")
axes[2].set_ylabel(r"$\mathcal{A}_{2,2,1,+}$")
axes[3].set_ylabel(r"$\mathcal{A}_{2,2,2,+}$")
axes[4].set_ylabel(r"$\mathcal{A}_{2,2,3,+}$")
axes[4].set_xlabel(r"$t_0$ \, [M]")

plt.tight_layout()
plt.subplots_adjust(right=1)
fig.savefig("outputs/violin_12", bbox_inches="tight") 
plt.close(fig)
