import numpy as np
import matplotlib.pyplot as plt
import bgp_qnm_fits as bgp

from matplotlib.colors import LinearSegmentedColormap
from plot_config import PlotConfig
from matplotlib.gridspec import GridSpec

config = PlotConfig()
config.apply_style()

custom_colormap = LinearSegmentedColormap.from_list("custom_colormap2", config.colors2)
colors = custom_colormap(np.linspace(0, 1, 3))

SIMNUMS = [
    "0001",
    "0002",
    "0003",
    "0004",
    "0005",
    "0006",
    "0007",
    "0008",
    "0009",
    "0010",
    "0011",
    "0012",
    "0013",
]
TRAINING_SPH_MODES = [
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
    (2, -2),
    (2, -1),
    (3, -3),
    (3, -2),
    (4, -4),
]

SIM_TRAINING_MODE_RULES = {
    "0001": "PE",
    "0002": "PE",
    "0003": "PE",
    "0004": "PE",
    "0005": "P",
    "0006": "P",
    "0007": "P",
    "0008": "ALL",
    "0009": "E",
    "0010": "P",
    "0011": "P",
    "0012": "P",
    "0013": "ALL",
}

TRAINING_START_TIME = -10
TRAINING_END_TIME = 100
TIME_STEP = 0.1

mode_filters = {
    "PE": lambda mode: mode[1] >= 0 and mode[1] % 2 == 0,
    "P": lambda mode: mode[1] >= 0,
    "E": lambda mode: mode[1] % 2 == 0,
    "ALL": lambda mode: True,
}

analysis_times = np.arange(TRAINING_START_TIME, TRAINING_END_TIME, TIME_STEP)

tuned_param_dict_GP = bgp.get_param_data("GP")
tuned_param_dict_WN = bgp.get_param_data("WN")
tuned_param_dict_GPC = bgp.get_param_data("GPc")


def kl_divergence_figs():

    sn_list_full = np.array([])
    cn_list_full = np.array([])
    sc_list_full = np.array([])

    for sim_id, mode_rule in SIM_TRAINING_MODE_RULES.items():  # Only want one figure

        sn_list = np.array([])
        cn_list = np.array([])
        sc_list = np.array([])

        spherical_modes = tuned_param_dict_WN[sim_id].keys()
        spherical_modes = [mode for mode in spherical_modes if mode[1] != 0]

        spherical_mode_choice = [mode for mode in TRAINING_SPH_MODES if mode_filters[mode_rule](mode)]

        for sph_mode in spherical_mode_choice:

            kernel_matrix_WN = bgp.compute_kernel_matrix(
                analysis_times, tuned_param_dict_WN[sim_id][sph_mode], bgp.kernel_WN
            )
            kernel_matrix_GP = bgp.compute_kernel_matrix(
                analysis_times, tuned_param_dict_GP[sim_id][sph_mode], bgp.kernel_GP
            )
            kernel_matrix_GPC = bgp.compute_kernel_matrix(
                analysis_times, tuned_param_dict_GPC[sim_id][sph_mode], bgp.kernel_GPC
            )

            kl_div_sn = bgp.js_divergence(kernel_matrix_WN, kernel_matrix_GP)
            kl_div_cn = bgp.js_divergence(kernel_matrix_GP, kernel_matrix_GPC)
            kl_div_sc = bgp.js_divergence(kernel_matrix_WN, kernel_matrix_GPC)

            sn_list = np.append(sn_list, np.log10(kl_div_sn))
            cn_list = np.append(cn_list, np.log10(kl_div_cn))
            sc_list = np.append(sc_list, np.log10(kl_div_sc))

        sn_list[sn_list == -np.inf] = np.nan
        cn_list[cn_list == -np.inf] = np.nan
        sc_list[sc_list == -np.inf] = np.nan

        sn_list_full = np.append(sn_list_full, sn_list)
        cn_list_full = np.append(cn_list_full, cn_list)
        sc_list_full = np.append(sc_list_full, sc_list)

        if sim_id != "0005":
            continue

        fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))

        spherical_mode_choice_labels = [f"$({mode[0]}, {mode[1]})$" for mode in spherical_mode_choice]

        ax.scatter(
            spherical_mode_choice_labels,
            sc_list,
            label="WN, GPc",
            color=colors[0],
            alpha=1,
        )
        ax.scatter(
            spherical_mode_choice_labels,
            sn_list,
            label="WN, GP",
            facecolors="none",
            edgecolors=colors[1],
            linewidth=1.5,
            alpha=1,
        )
        ax.scatter(
            spherical_mode_choice_labels,
            cn_list,
            label="GP, GPc",
            color=colors[2],
            alpha=1,
        )

        ax.set_xlabel("Spherical Mode")
        ax.set_ylabel(r"$\log_{10}(D)$")

        ax.tick_params(axis="x", labelsize=8)

        ax.legend(loc="upper center", frameon=True, framealpha=0.7, bbox_to_anchor=(0.5, 0.9))

        ax.yaxis.grid(False)
        ax.xaxis.grid(True, linestyle="-", alpha=0.8)

        # fig.savefig(f"outputs/JS_{sim_id}.pdf")
        # plt.close(fig)

    return sn_list_full, cn_list_full, sc_list_full


def kl_divergence_histogram(sn_list_full, cn_list_full, sc_list_full):
    fig = plt.figure(figsize=(config.fig_width, config.fig_height))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.04)  # Adjust hspace for overlap

    bins = np.linspace(7.32, 7.36, 12)

    y_min = 0
    y_max = 22

    ax1 = fig.add_subplot(gs[0])
    ax1.hist(
        cn_list_full,
        alpha=0.7,
        label="GP, GPc",
        color=colors[2],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xlim(-4.8, -3.6)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xticks(np.arange(-4.8, -3.6, 0.5))
    ax1.spines["right"].set_visible(False)  # Hide right spine
    ax1.tick_params(axis="x", which="both", labelbottom=True)
    ax1.tick_params(axis="y", which="both", right=False, left=False, labelleft=False)
    ax1.set_ylabel("Frequency")
    ax1.legend(bbox_to_anchor=(1, 1), loc="upper right", frameon=False)

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(
        sc_list_full,
        bins=bins - 0.005,
        alpha=0.7,
        label="WN, GPc",
        color=colors[0],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.hist(
        sn_list_full,
        bins=bins + 0.005,
        alpha=0.7,
        label="WN, GP",
        color=colors[1],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xlim(7.32, 7.365)  # Set x-axis limits for the second range
    ax2.set_ylim(y_min, y_max)
    ax2.set_xticks(np.arange(7.33, 7.36, 0.02))
    ax2.spines["left"].set_visible(False)  # Hide left spine
    fig.text(0.5, 0.005, r"$\log_{10}(D)$", ha="center", va="center")
    ax2.tick_params(axis="x", which="both", labelbottom=True)
    ax2.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax2.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.05, 1))

    d = 0.015  # Size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)

    # Top diagonal lines
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    ax2.plot(((1 - d) + 0.04, (1 + d) + 0.04), (1 - d, 1 + d), **kwargs)
    ax2.plot(((1 - d) + 0.04, (1 + d) + 0.04), (-d, +d), **kwargs)

    fig.savefig("outputs/JS_histogram.pdf")


if __name__ == "__main__":
    sn_list_full, cn_list_full, sc_list_full = kl_divergence_figs()
    kl_divergence_histogram(sn_list_full, cn_list_full, sc_list_full)
