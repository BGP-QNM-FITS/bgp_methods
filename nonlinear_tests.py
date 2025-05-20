import qnmfits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from CCE import SXS_CCE
import bgp_qnm_fits as bgp

from scipy.optimize import minimize

from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot_config import PlotConfig
from scipy.stats import linregress

id = "0001"
N_MAX=7
T=100
T0_REF=17
num_samples=10000
include_Mf=True
include_chif=True
sim_main = SXS_CCE(id, lev="Lev5", radius="R2")
qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)]
spherical_modes = [(2, 2)]
chif_mag_ref = sim_main.chif_mag
Mf_ref = sim_main.Mf

tuned_param_dict_GP = bgp.get_param_data("GP")[id]
tuned_param_dict_WN = bgp.get_param_data("WN")[id]
tuned_param_dict_GPC = bgp.get_param_data("GPc")[id]

ref_fit_GP = bgp.BGP_fit(
    sim_main.times,
    sim_main.h,
    qnm_list,
    Mf_ref,
    chif_mag_ref,
    tuned_param_dict_GP,
    bgp.kernel_main,
    t0=T0_REF,
    use_nonlinear_params=False,
    num_samples=num_samples,
    t0_method="geq",
    T=T,
    spherical_modes=spherical_modes,
    include_chif=include_chif,
    include_Mf=include_Mf,
)

samples = ref_fit_GP.fit["samples"] 

mass_samples = samples[:, -1]
spin_samples = samples[:, -2]

# Fit a line through mass and spin samples

# Perform linear regression
result = linregress(spin_samples, mass_samples)
slope = result.slope
intercept = result.intercept
r_squared = result.rvalue**2

chifs = np.linspace(0.65, 0.72, 5)
Mfs = slope * chifs + intercept

fits = [] 
for chif, Mf in zip(chifs, Mfs):
    print(chif)
    fit = bgp.BGP_fit(
        sim_main.times,
        sim_main.h,
        qnm_list,
        Mf,
        chif,
        tuned_param_dict_GP,
        bgp.kernel_main,
        t0=T0_REF,
        use_nonlinear_params=False,
        num_samples=num_samples,
        t0_method="geq",
        T=T,
        spherical_modes=spherical_modes,
        include_chif=include_chif,
        include_Mf=include_Mf,
    )
    fits.append(fit)

colors = plt.cm.viridis(np.linspace(0, 1, len(fits))) 
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

for j, fit in enumerate(fits):

    print(fit) 

    ref_fit_GP_NL_samples = fit.fit["samples"]

    spins_GP_NL = ref_fit_GP_NL_samples[:, -2]
    masses_GP_NL = ref_fit_GP_NL_samples[:, -1]

    real_freq_GP_deviations_NL = []
    imag_freq_GP_deviations_NL = []
    mixing_GP_deviations_NL = []

    ell, m, n, sign = (2, 2, 0, 1)
    ellp, mp = (2, 2)

    ref_freq = qnmfits.qnm.omega(ell, m, n, sign, chif_mag_ref, Mf=Mf_ref, s=-2)
    ref_mixing = qnmfits.qnm.mu(ellp, mp, ell, m, n, sign, chif_mag_ref)

    for i, _ in enumerate(ref_fit_GP_NL_samples):

        approx_freq_GP_NL = qnmfits.qnm.omega(ell, m, n, sign, spins_GP_NL[i], Mf=masses_GP_NL[i], s=-2)
        approx_mixing_GP_NL = qnmfits.qnm.mu(ellp, mp, ell, m, n, sign, spins_GP_NL[i])

        linear_approx_freq = ref_freq + (spins_GP_NL[i] - chif_mag_ref) * 
        
        (masses_GP_NL[i] - Mf_ref) * )

        real_freq_GP_deviations_NL.append(np.abs(np.real(approx_freq_GP_NL - linear_approx_freq)) / np.real(ref_freq))
        imag_freq_GP_deviations_NL.append(np.abs(np.imag(approx_freq_GP_NL - linear_approx_freq)) / -np.imag(ref_freq))

        mixing_GP_deviations_NL.append(np.abs(approx_mixing_GP_NL - linear_approx_mixing) / np.abs(ref_mixing))

    real_freq_GP_deviations_NL = np.array(real_freq_GP_deviations_NL)
    imag_freq_GP_deviations_NL = np.array(imag_freq_GP_deviations_NL)
    mixing_GP_deviations_NL = np.array(mixing_GP_deviations_NL)

    real_freq_GP_deviations_NL = np.hstack((real_freq_GP_deviations_NL, -real_freq_GP_deviations_NL))
    imag_freq_GP_deviations_NL = np.hstack((imag_freq_GP_deviations_NL, -imag_freq_GP_deviations_NL))
    mixing_GP_deviations_NL = np.hstack((mixing_GP_deviations_NL, -mixing_GP_deviations_NL))

    sns.kdeplot(
        real_freq_GP_deviations_NL,
        ax=ax,
        color=colors[j],
        label=r"$\rm Re(\omega_{\alpha})$ (NL)",
        lw=1,
        linestyle="--",
    )
    sns.kdeplot(
        imag_freq_GP_deviations_NL,
        ax=ax,
        color=colors[j],
        label=r"$\rm Im(\omega_{\alpha})$ (NL)",
        lw=1,
        linestyle=":",
    )
    sns.kdeplot(mixing_GP_deviations_NL, ax=ax, color=colors[j], lw=1, linestyle="-", label=r"$\mu_{\alpha}$ (NL)")

ax.set_xlabel("Fractional deviation")
ax.set_xscale("log")
ax.set_yscale("log")
# Create a colormap for the chif values
norm = plt.Normalize(chifs.min(), chifs.max())
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])

# Add colorbar
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r'$\chi_f$')

# Add legend for line styles
line_styles = [
    Line2D([0], [0], color='gray', lw=0.5, linestyle='-', label=r'$\mu_{\alpha}$'),
    Line2D([0], [0], color='gray', lw=0.5, linestyle='--', label=r'$\rm Re(\omega_{\alpha})$'),
    Line2D([0], [0], color='gray', lw=0.5, linestyle=':', label=r'$\rm Im(\omega_{\alpha})$')
]
ax.legend(handles=line_styles, loc='best')
# ax.set_xlim(1e-4, 2e-2)
ax.set_ylim(1e-1, 1e4)
plt.show() 