import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import jax.numpy as jnp
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LinearSegmentedColormap
from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()

def smoothmax(x, x_max, smoothness):
    return (x + x_max - np.sqrt((x - x_max) ** 2 + smoothness*x_max**2)) * 0.5

x = np.linspace(-5, 100, 10000)
smoothness = np.logspace(-3, -1, 10)
x_max = 1

custom_colormap = LinearSegmentedColormap.from_list("custom_colormap2", config.colors2)
colors = custom_colormap(np.linspace(0, 1, len(smoothness)))

fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))

for i, smoothness in enumerate(smoothness):
    ax.plot(x, smoothmax(x, x_max, smoothness[i]), color=colors[i])

ax.plot(x, np.clip(x, 0, x_max), c="k", label=r"max($x$, $x_{\rm max}$)", ls="--")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\textsc{SmoothMax}(x)$")
ax.set_xlim(0, 2)
ax.set_ylim(0, 1.15)
ax.set_aspect("equal")
ax.axhline(x_max, c="k", alpha=0.5, lw=1, ls="--")
# ax.text(1.71, x_min + 0.08, r"$x_{\rm min}$", va="center", ha="left", c="k", alpha=0.5)
ax.text(0.05, x_max + 0.08, r"$x_{\rm max}$", va="center", ha="left", c="k", alpha=0.5)
ax.legend(frameon=False, loc="lower right")

norm = mcolors.LogNorm(vmin=smoothness.min(), vmax=smoothness.max())
sm = plt.cm.ScalarMappable(cmap=custom_colormap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, format=LogFormatter(), shrink=0.6)
cbar.set_label(r"$s$", labelpad=10, rotation=0)

fig.savefig("outputs/smoothmax.pdf", dpi=600, bbox_inches="tight")
