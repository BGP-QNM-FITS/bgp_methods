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
NUM_SAMPLES = 1000000
INCLUDE_MF = True
INCLUDE_CHIF = True

sim_main = bgp.SXS_CCE(ID, type=DATA_TYPE, lev="Lev5", radius="R2") 
Mf_ref = sim_main.Mf
chif_mag_ref = sim_main.chif_mag
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[ID]

qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)] + [(3,2,0,1)]
spherical_modes = [(2, 2)]

T0 = 17 
T = 100 

HYPERPARAM_RULE_DICT_GP = {
    "sigma_max": "multiply",
    "period": "multiply",
}

variations = np.linspace(0.001, 1, 100) 
hyperparam_variations = [[1, val] for val in variations]

samples_list = np.zeros((len(hyperparam_variations), NUM_SAMPLES, len(qnm_list) * 2 + 2))

for i, hyperparams in enumerate(hyperparam_variations): 

    # slight abuse of this function but its actually flexible enough to do this i think and hope  
    tuned_params_new = bgp.get_tuned_params(tuned_param_dict_GP, hyperparams, HYPERPARAM_RULE_DICT_GP)

    fit_GP = bgp.BGP_fit(
                sim_main.times,
                sim_main.h,
                qnm_list,
                Mf_ref,
                chif_mag_ref,
                tuned_params_new,
                bgp.kernel_GP,
                t0=T0,
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

    samples_list[i, :, :] = fit_GP.fit["samples"]


# Set up the plot
plt.figure(figsize=(12, 10))

# Create a colormap for different hyperparam variations
colors = plt.cm.viridis(np.linspace(0, 1, len(hyperparam_variations)))

# Create a legend elements list
legend_elements = []

# Plot each variation with a different color
for i, (hyperparams, color) in enumerate(zip(hyperparam_variations, colors)):
    # Extract samples for this hyperparam variation
    samples = samples_list[i, :, 2:4]
    
    # Create a corner plot with this color
    fig = corner.corner(
        samples, 
        color=to_hex(color),
        show_titles=True,
        title_fmt=".3f",
        plot_datapoints=False,
        plot_density=True,
        levels=(0.68, 0.95),
        fig=plt.gcf()
    )
    
    # Add to legend
    legend_elements.append(Line2D([0], [0], color=to_hex(color), lw=2, 
                            label=f"$\\mu_{{max}}$ = {hyperparams[1]:.1f}"))

# Add a legend outside the plot
plt.legend(handles=legend_elements, 
            loc='upper right', 
            bbox_to_anchor=(0.95, 0.95),
            title="Hyperparameter Variations")

plt.tight_layout()
plt.savefig("outputs/hyperparam_variation.png")


# Calculate the 90% width of the posterior distributions
posterior_widths = np.zeros((len(hyperparam_variations), 2))  # For Mf and chif

for i in range(len(hyperparam_variations)):
    samples = samples_list[i, :, 0:2]
    
    # Calculate the 5th and 95th percentiles for each parameter
    q05, q95 = np.percentile(samples, [5, 95], axis=0)
    
    # Store the widths (difference between 95th and 5th percentiles)
    posterior_widths[i, :] = q95 - q05

# Create a new figure for the posterior width vs hyperparameter plot
plt.figure(figsize=(10, 6))

# Extract the hyperparam values for x-axis
hyperparam_values = [h[1] for h in hyperparam_variations]

# Plot width vs hyperparameter value for both parameters
plt.plot(hyperparam_values, posterior_widths[:, 0], 'o-', label='$Re C$ posterior width')
plt.plot(hyperparam_values, posterior_widths[:, 1], 's-', label='$Im C$ posterior width')

plt.xlabel('$\mu_{max}$ value')
plt.ylabel('90% Credible Interval Width')
plt.title('Posterior Width vs Hyperparameter Value')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/hyperparam_vs_posterior_width.png")