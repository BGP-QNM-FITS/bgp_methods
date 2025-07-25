import numpy as np
import matplotlib.pyplot as plt
import bgp_qnm_fits as bgp

from matplotlib.colors import LinearSegmentedColormap
from plot_config import PlotConfig
from matplotlib.lines import Line2D

config = PlotConfig()
#config.apply_style()

id = "0001"

TRAINING_START_TIME = -10
TRAINING_END_TIME = 310
TIME_STEP = 0.1

analysis_times = np.arange(TRAINING_START_TIME, TRAINING_START_TIME + TRAINING_END_TIME, TIME_STEP)

data_type = 'strain'

R = bgp.get_residual_data(big=True, data_type=data_type)[id]
tuned_params = bgp.get_tuned_param_dict("GP", data_type=data_type)[id]
param_dict_lm = bgp.get_param_dict(data_type=data_type)[id] 

HYPERPARAM_RULE_DICT_GP = {
    "sigma_max": "multiply",
    "period": "multiply",
}

#hyperparams = [6.1930717137040645, 0.38265931880613024]
#tuned_params = bgp.get_tuned_params(param_dict_lm, hyperparams, HYPERPARAM_RULE_DICT_GP, spherical_modes=None)

spherical_modes = [(2, 2)]
#spherical_modes = [(2,1), (3, 2), (4, 4)]

kernel_dict = {
    mode: bgp.compute_kernel_matrix(analysis_times, tuned_params[mode], bgp.kernel_GP) for mode in spherical_modes
}

fig, axs = plt.subplots(1, 1, sharex=True, figsize=(config.fig_width, config.fig_height))

custom_colormap = LinearSegmentedColormap.from_list("custom_colormap2", config.colors2)
colors = custom_colormap(np.linspace(0, 1, 3))

for i, (ell, m) in enumerate(spherical_modes):

    axs.plot(
        analysis_times,
        np.abs(np.real(R[ell, m])),
        color=colors[i],
    )
    #axs.plot(
    #    analysis_times,
    #    np.abs(np.imag(R[ell, m])),
    #    color=colors[i],
    #    ls=":",
    #)

    axs.set_title(rf"$(\ell, m) = ({ell}, {m})$")
    axs.set_xlim(-10, 300)
    axs.set_ylabel(r"$\mathfrak{r} \,\, [M]$")
    axs.set_yscale("log")
    #axs[i].set_ylim(1e-11, 3e-3)
    axs.set_ylim(1e-11, 1e-2)

solid_line = Line2D([0], [0], color="black", linestyle="-")
dotted_line = Line2D([0], [0], color="black", linestyle=":")

line_legend = axs.legend(
    [solid_line, dotted_line],
    [r"$\rm Re(\mathfrak{r}^{\beta}_{i})$", r"$\rm Im(\mathfrak{r}^{\beta}_{i})$"],
    frameon=False,
    ncol=2,
    loc="lower left",
    bbox_to_anchor=(0.05, -0.75),
    columnspacing=0.9,
    handletextpad=0.5,
)
axs.add_artist(line_legend)
axs.legend(frameon=False, loc="lower right", ncol=1, bbox_to_anchor=(0.93, -0.25))
axs.set_xlabel("$t \,\, [M]$")

fig.savefig(f"outputs/credible_regions_log_{data_type}_pres.pdf", dpi=600, bbox_inches="tight")
