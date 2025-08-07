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

N_MAX = 7
ID = "0001"
DATA_TYPE = "strain"
NUM_SAMPLES = 1000000
INCLUDE_MF = True
INCLUDE_CHIF = True

sim_main = bgp.SXS_CCE(ID, type=DATA_TYPE, lev="Lev5", radius="R2") 
Mf_ref = sim_main.Mf
chif_mag_ref = sim_main.chif_mag
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[ID]
qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)] + [(3,2,0,1)]
spherical_modes = [(2, 2)]

T0s = np.linspace(-100, 10, 110)
T = 100 

N_effs = [] 

significances_GP = np.zeros((len(T0s), len(qnm_list)))

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
    significances_GP[i, :] = bgp.get_significance_list(
                    qnm_list,
                    np.array(fit["mean"]),
                    np.array(fit["fisher_matrix"]),
                    include_chif=INCLUDE_CHIF,
                    include_Mf=INCLUDE_MF,
                )

for qnm in range(len(qnm_list)):
    plt.plot(T0s, significances_GP[:, qnm], label=f"Mode {qnm_list[qnm]}")

plt.savefig("outputs/early_time_significance.png")