import numpy as np
import matplotlib.pyplot as plt
import bgp_qnm_fits as bgp
from matplotlib.colors import LinearSegmentedColormap
from plot_config import PlotConfig
from matplotlib.lines import Line2D

config = PlotConfig()
config.apply_style()

id = "0001"
sim = bgp.SXS_CCE(id, type='strain', lev="Lev5", radius="R2")
M = sim.Mf
mode = (2, 2)
TRAINING_START_TIME = -10
TRAINING_END_TIME = 310
TIME_STEP = 0.1
analysis_times = np.arange(TRAINING_START_TIME, TRAINING_START_TIME + TRAINING_END_TIME, TIME_STEP)

data_types = ["strain", "news", "psi4"]
labels = [r"$h$", r"$M\mathcal{N}$", r"$M^{2}\Psi_4$"]

fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
colors = LinearSegmentedColormap.from_list("custom_colormap", config.colors2)(np.linspace(0, 1, len(data_types)))

for idx, data_type in enumerate(data_types):
    R = bgp.get_residual_data(big=True, data_type=data_type)[id]
    
    scaling_factor = 1.0
    if data_type == "news":
        scaling_factor = M
    elif data_type == "psi4":
        scaling_factor = M**2  
    
    ax.semilogy(
        analysis_times,
        np.abs(np.real(R[mode])) * scaling_factor,
        color=colors[idx],
        label=labels[idx],
    )

ax.set_xlabel("$t \,\, [M]$")
ax.set_ylabel(r"$\mathfrak{r}^{(2,2)}_{0001}$")
ax.set_xlim(TRAINING_START_TIME, TRAINING_START_TIME + TRAINING_END_TIME)
ax.set_ylim(1e-15, 1e-2)

type_handles = [Line2D([0], [0], color=colors[i]) for i in range(len(data_types))]

ax.legend(
    type_handles,
    labels,
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    ncol=1
)

plt.savefig("outputs/residual_comparison.pdf", dpi=600, bbox_inches="tight")
