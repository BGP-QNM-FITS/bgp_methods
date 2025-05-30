import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LinearSegmentedColormap
from plot_config import PlotConfig
from matplotlib.lines import Line2D

config = PlotConfig()
config.apply_style()

import bgp_qnm_fits as bgp

id = "0001"

TRAINING_START_TIME = -10
TRAINING_END_TIME = 100
TIME_STEP = 0.1

analysis_times = np.arange(TRAINING_START_TIME, TRAINING_START_TIME + TRAINING_END_TIME, TIME_STEP)

R = bgp.get_residual_data()[id]
#tuned_params = bgp.get_param_data("GP")[id]

HYPERPARAM_RULE_DICT_GP = {
    "sigma_max": "multiply",
    "t_s": "sum",
    "length_scale": "multiply",
    "period": "multiply",
}
param_dict = bgp.get_param_dict()[id]
hyperparams = [0.7, 17.86536401, 1.13781152, 0.2583172]

tuned_params = bgp.get_tuned_params(param_dict, hyperparams, HYPERPARAM_RULE_DICT_GP, spherical_modes=None)

spherical_modes = [(2, 2), (3, 2), (4, 4)]

kernel_dict = {
    mode: bgp.compute_kernel_matrix(analysis_times, tuned_params[mode], bgp.kernel_main) for mode in spherical_modes
}

fig, axs = plt.subplots(len(spherical_modes), 1, sharex=True, figsize=(config.fig_width, config.fig_height * 2))

custom_colormap = LinearSegmentedColormap.from_list("custom_colormap2", config.colors2)
colors = custom_colormap(np.linspace(0, 1, 3))

for i, (ell, m) in enumerate(spherical_modes):

    axs[i].plot(
        analysis_times,
        np.real(R[ell, m]),
        color=colors[i],
    )
    axs[i].plot(
        analysis_times,
        np.imag(R[ell, m]),
        color=colors[i],
        ls=":",
    )

    # Plot a line segment to indicate the size of 1 x axis unit

    period_length = tuned_params[(ell, m)]["period"]

    y_pos = np.min(np.real(R[ell, m])) + 0.7 * np.ptp(np.real(R[ell, m]))

    axs[i].plot(
        [40, 40 + period_length],
        [y_pos, y_pos],
        "k-",
    )
    axs[i].plot(
        [39.8, 39.8],
        [
            y_pos - 0.03 * np.ptp(np.real(R[ell, m])),
            y_pos + 0.03 * np.ptp(np.real(R[ell, m])),
        ],
        "k-",
    )
    axs[i].plot(
        [40.2 + period_length, 40.2 + period_length],
        [
            y_pos - 0.03 * np.ptp(np.real(R[ell, m])),
            y_pos + 0.03 * np.ptp(np.real(R[ell, m])),
        ],
        "k-",
    )
    axs[i].text(
        40 + period_length / 2,
        y_pos + 0.14 * np.ptp(np.real(R[ell, m])),
        r"$P^{\beta}_i$",
        ha="center",
        fontsize=7,
    )

    axs[i].fill_between(
        analysis_times,
        -np.sqrt(np.diag(kernel_dict[ell, m])),
        np.sqrt(np.diag(kernel_dict[ell, m])),
        color="k",
        alpha=0.2,
        label=r"1$\sigma$",
    )

    axs[i].set_title(rf"$\beta = ({ell}, {m})$")
    #axs[i].set_xlim(0, 90)
    axs[i].set_ylabel(r"$\mathfrak{r}^{\beta}_{i} \,\, [M]$")

solid_line = Line2D([0], [0], color="black", linestyle="-")
dotted_line = Line2D([0], [0], color="black", linestyle=":")

line_legend = axs[-1].legend(
    [solid_line, dotted_line],
    [r"$\rm Re(\mathfrak{r}^{\beta}_{i})$", r"$\rm Im(\mathfrak{r}^{\beta}_{i})$"],
    frameon=False,
    ncol=2,
    loc="lower left",
    bbox_to_anchor=(0.05, -0.75),
    columnspacing=0.9,
    handletextpad=0.5,
)
axs[-1].add_artist(line_legend)
axs[-1].legend(frameon=False, loc="lower right", ncol=1, bbox_to_anchor=(0.93, -0.75))
axs[-1].set_xlabel("$t \,\, [M]$")

fig.savefig("outputs/credible_regions.pdf", dpi=600, bbox_inches="tight")
