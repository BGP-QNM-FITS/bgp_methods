import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import bgp_qnm_fits as bgp

from matplotlib.colors import to_hex
from matplotlib.colors import LinearSegmentedColormap

from plot_config import PlotConfig
from scipy.interpolate import interp1d

config = PlotConfig()
config.apply_style()

ID="0001"
N_MAX=6
T0 = 17
T=100

data_types = ['strain', 'news', 'psi4']

fig, axes = plt.subplots(6, 3, figsize=(config.fig_width * 2, config.fig_height * 4))
axes = axes.flatten()
fig.subplots_adjust(wspace=0, hspace=0.5)

qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)] + [(3,2,0,1)]
spherical_modes = [(2, 2)]

colors = config.colors
colors2 = config.colors2
custom_colormap = LinearSegmentedColormap.from_list("custom_colormap", colors)
custom_colormap2 = LinearSegmentedColormap.from_list("custom_colormap2", colors2)

colors_2 = custom_colormap2(np.linspace(0, 1, 3))

for i, data_type in enumerate(data_types):

    sim_main = bgp.SXS_CCE(ID, type=data_type, lev="Lev5", radius="R2")
    chif_mag_ref = sim_main.chif_mag
    Mf_ref = sim_main.Mf

    tuned_param_dict_WN = bgp.get_tuned_param_dict("WN", data_type=data_type)[ID]
    tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=data_type)[ID]

    fit_WN = bgp.BGP_fit(
        sim_main.times,
        sim_main.h,
        qnm_list,
        Mf_ref,
        chif_mag_ref,
        tuned_param_dict_WN,
        bgp.kernel_WN,
        t0=T0,
        num_samples=10000,
        t0_method="geq",
        T=T,
        spherical_modes=spherical_modes,
        include_chif=True,
        include_Mf=True,
        strain_parameters=True,
        data_type=data_type,
    )

    fit_GP = bgp.BGP_fit(
        sim_main.times,
        sim_main.h,
        qnm_list,
        Mf_ref,
        chif_mag_ref,
        tuned_param_dict_GP,
        bgp.kernel_GP,
        t0=T0,
        num_samples=10000,
        t0_method="geq",
        T=T,
        spherical_modes=spherical_modes,
        include_chif=True,
        include_Mf=True,
        strain_parameters=True,
        data_type=data_type,
    )

    samples_WN = fit_WN.fit["samples"]
    samples_GP = fit_GP.fit["samples"]

    if data_type == 'strain':
        ref_samples = samples_GP

    for j in range(len(qnm_list) * 2 + 2):
        param_samples_WN = samples_WN[:, j]
        param_samples_GP = samples_GP[:, j]

        #sns.kdeplot(param_samples_WN, ax=axes[j], color=colors_2[i], bw_adjust=1, ls='--', label=f"{data_type}")
        sns.kdeplot(param_samples_GP, ax=axes[j], color=colors_2[i], bw_adjust=2, ls='-')

for j in range(len(qnm_list) * 2 + 2):
    param_names = fit_GP.fit["param_names"]
    ref_params = fit_GP.fit["ref_params"]
    ref_params_nonlinear = fit_GP.fit["ref_params_nonlinear"]
    axes[j].set_ylabel("")
    axes[j].set_yticks([])
    if j < len(qnm_list) * 2:
        n = j // 2
        component = "Re" if j % 2 == 0 else "Im"
        title = fr"{component}$\, C_{{(2, 2, {n}, +)}} \,\, [M]$"
        if n == 7:
            title = fr"{component}$\, C_{{(3, 2, {0}, +)}} \,\, [M]$"
        axes[j].set_title(title)
        axes[j].axvline(ref_params[j], color="k", ls='--', alpha=0.3)
        axes[j].axvline(ref_params_nonlinear[j], color="k", ls='-', alpha=0.3)
    elif j == len(qnm_list) * 2:
        axes[j].set_title(r"$\chi_f$")
    elif j == len(qnm_list) * 2 + 1:
        axes[j].set_title(r"$M_f \,\, [M]$")
    lower_bound, upper_bound = np.percentile(ref_samples[:, j], [10, 90])
    axes[j].set_xlim(lower_bound, upper_bound)
    axes[j].set_xticks(np.round(np.linspace(lower_bound, upper_bound, 3), 3))

chif_nonlinear, Mf_nonlinear = fit_GP.get_nonlinear_mf_chif(T0, fit_GP.T, fit_GP.spherical_modes, fit_GP.chif_ref, fit_GP.Mf_ref)

axes[-1].axvline(sim_main.Mf, color="k", ls='--', alpha=0.3)
axes[-1].axvline(Mf_nonlinear, color="k", ls='-', alpha=0.3)

axes[-2].axvline(sim_main.chif_mag, color="k", ls='--', alpha=0.3)
axes[-2].axvline(chif_nonlinear, color="k", ls='-', alpha=0.3)

# Create legend for linestyles
linestyle_handles = [
    plt.Line2D([0], [0], color="black", ls='--', label="WN"),
    plt.Line2D([0], [0], color="black", ls='-', label="GP")
]
#axes[-3].legend(handles=linestyle_handles, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2, frameon=False)

# Create legend for colors
color_handles = [
    plt.Line2D([0], [0], color=colors_2[0], label=r"$h$"),
    plt.Line2D([0], [0], color=colors_2[1], label=r"$\mathfrak{N}$"),
    plt.Line2D([0], [0], color=colors_2[2], label=r"$\Psi_4$")
]

fig.legend(handles=color_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
fig.savefig("outputs/strain_news_psi4.pdf", dpi=600, bbox_inches="tight")
