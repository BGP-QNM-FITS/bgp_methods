import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import jax.numpy as jnp
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LinearSegmentedColormap
from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()


def logoneplusexp(x):
    ans = np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)
    return ans


def smoothclip(x, x_max, sharpness):
    clipped_x = x - (1.0 / sharpness) * logoneplusexp(sharpness * (x - x_max))
    return clipped_x


def softclip(x, x_max, sharpness):
    return np.exp(smoothclip(np.log(x), np.log(x_max), sharpness))


def smoothmax(x, x_max, sharpness):
    return (x + x_max - np.sqrt((x - x_max) ** 2 + sharpness*x_max**2)) * 0.5


def exponential_func(x, length_scale, t_s, sigma_max):
    return sigma_max * jnp.exp(-(x - t_s) / length_scale)

def new_func(x, length_scale, t_s, sigma_max, sharpness):
    return smoothmax(
        exponential_func(x, length_scale, t_s, sigma_max),
        sigma_max,
        sharpness,
    )


def old_func(x, length_scale, t_s, sigma_max, sharpness):
    return softclip(
        exponential_func(x, length_scale, t_s, sigma_max),
        sigma_max,
        sharpness,
    )

x = np.linspace(-5, 100, 10000)
# x_clip = np.linspace(0, 2, 10000)
sharpnesses1 = np.logspace(-2, -1, 10)
sharpnesses2 = np.logspace(0, 2, 10)
x_max = 1e-3
length_scale = 10
ts = 30

custom_colormap = LinearSegmentedColormap.from_list("custom_colormap2", config.colors2)
colors = custom_colormap(np.linspace(0, 1, len(sharpnesses1)))

fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))

vals = new_func(x, length_scale, ts, x_max, 1e-3)
vals2 = old_func(x, length_scale, ts, x_max, 1)

# ax.plot(x, x, ls=":", c="k")
for i, sharpness in enumerate(sharpnesses1):
    ax.plot(x, old_func(x, length_scale, 34, x_max, sharpnesses2[i]), color=colors[i], ls="--")
    ax.plot(x, new_func(x, length_scale, 34, x_max, 5e-2), color=colors[i])

# ax.plot(x_clip, np.clip(x_clip, 0, x_max), c="k", label="np.clip", ls="--")
ax.set_xlabel("$x$")
# ax.set_ylabel(r"$\textsc{SoftClip}(x)$")
#ax.set_xlim(-0.2, 1.5)
#ax.set_ylim(-2, 1.2)
#ax.set_ylim(0.5, 1.2)
# ax.set_aspect("equal")
# ax.axhline(x_min, c="k", alpha=0.5, lw=1, ls="--")
# ax.axhline(x_max, c="k", alpha=0.5, lw=1, ls="--")
# ax.text(1.71, x_min + 0.08, r"$x_{\rm min}$", va="center", ha="left", c="k", alpha=0.5)
#ax.text(0, x_max + 0.1, r"$x_{\rm max}$", va="center", ha="left", c="k", alpha=0.5)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.055))

norm = mcolors.LogNorm(vmin=sharpnesses1.min(), vmax=sharpnesses1.max())
sm = plt.cm.ScalarMappable(cmap=custom_colormap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, format=LogFormatter(), shrink=0.6)
cbar.set_label(r"$s$", labelpad=10, rotation=0)

fig.savefig("outputs/softclip.pdf", dpi=600, bbox_inches="tight")
