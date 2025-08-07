import qnmfits
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import HTMLWriter
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

import bgp_qnm_fits as bgp

from scipy.optimize import minimize

from plot_config import PlotConfig
import corner
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
from matplotlib.animation import PillowWriter

N_MAX = 6
ID = "0001"
DATA_TYPE = "strain"
NUM_SAMPLES = 10000
INCLUDE_MF = True
INCLUDE_CHIF = True

sim_main = bgp.SXS_CCE(ID, type=DATA_TYPE, lev="Lev5", radius="R2") 
Mf_ref = sim_main.Mf
chif_mag_ref = sim_main.chif_mag
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[ID]
qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)] + [(3,2,0,1)]
spherical_modes = [(2, 2)]

T0s = np.linspace(-50, 100, 150)
T = 100 

mass_samples = np.zeros((len(T0s), NUM_SAMPLES))
chif_samples = np.zeros((len(T0s), NUM_SAMPLES))

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
            decay_corrected=False,
            num_samples=NUM_SAMPLES,
            t0_method="closest",
            T=T,
            spherical_modes=spherical_modes,
            include_chif=INCLUDE_CHIF,
            include_Mf=INCLUDE_MF,
            data_type=DATA_TYPE,
            strain_parameters=False, 
        )

for i, fit in enumerate(fits_GP.fits):
    chif_samples[i, :] = fit["samples"][:, -2]
    mass_samples[i, :] = fit["samples"][:, -1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# chif_samples subplot
for i, T0 in enumerate(T0s):
    lower = np.percentile(chif_samples[i], 16)
    median = np.percentile(chif_samples[i], 50)
    upper = np.percentile(chif_samples[i], 84)
    ax1.fill_between([T0], [lower], [upper], color="tab:blue", alpha=0.3)
    ax1.plot([T0], [median], marker="o", color="tab:blue")

ax1.set_ylabel("chif_samples")
ax1.set_title("chif_samples credible interval vs T0")

# mass_samples subplot
for i, T0 in enumerate(T0s):
    lower = np.percentile(mass_samples[i], 16)
    median = np.percentile(mass_samples[i], 50)
    upper = np.percentile(mass_samples[i], 84)
    ax2.fill_between([T0], [lower], [upper], color="tab:orange", alpha=0.3)
    ax2.plot([T0], [median], marker="o", color="tab:orange")

ax1.axhline(y=chif_mag_ref, color="black", linestyle="--", label="Mf_ref")
ax2.axhline(y=Mf_ref, color="black", linestyle="--", label="chif_mag_ref")

ax1.axvline(17, color="k", linestyle="--", alpha=0.3)
ax2.axvline(17, color="k", linestyle="--", alpha=0.3)

ax2.set_xlabel("T0")
ax2.set_ylabel("mass_samples")
ax2.set_title("mass_samples credible interval vs T0")
plt.tight_layout()
plt.savefig("outputs/mass_spin.pdf")