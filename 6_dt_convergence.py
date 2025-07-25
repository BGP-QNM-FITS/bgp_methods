import numpy as np
import pickle
import matplotlib.pyplot as plt
import bgp_qnm_fits as bgp

from matplotlib.colors import LinearSegmentedColormap
from plot_config import PlotConfig

config = PlotConfig()
#config.apply_style()

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
RINGDOWN_START_TIMES = [
    17.0,
    21.0,
    23.0,
    26.0,
    17.0,
    17.0,
    17.0,
    11.0,
    29.0,
    16.0,
    12.0,
    17.0,
    6.0,
]
TRAINING_SPH_MODES = [
    (2, 2),
    #(2, 1),
    #(3, 3),
    #(3, 2),
    #(4, 4),
    #(2, -2),
    #(2, -1),
    #(3, -3),
    #(3, -2),
    #(4, -4),
]

SIM_TRAINING_MODE_RULES = {
    "0001": "PE",
    #"0002": "PE",
    #"0003": "PE",
    #"0004": "PE",
    #"0005": "P",
    #"0006": "P",
    #"0007": "P",
    #"0008": "ALL",
    #"0009": "E",
    #"0010": "P",
    #"0011": "P",
    #"0012": "P",
    #"0013": "ALL",
}

SMOOTHNESS = 1e-3
TIME_SHIFT = 0 

# These determine the parameter and training range but do not have to match `analysis times' used later.

RESIDUAL_BIG_START = -10
RESIDUAL_BIG_END = 310
TIME_STEP = 0.1

TRAINING_START_TIME_WN = 0
TRAINING_RANGE_WN = 80

TRAINING_START_TIME_GP = 20 
TRAINING_RANGE_GP = 60 

# Define training bounds

SIGMA_MAX_LOWER, SIGMA_MAX_UPPER = 0.01, 50
T_S_LOWER, T_S_UPPER = -20, 30
LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER = 0.1, 5
PERIOD_LOWER, PERIOD_UPPER = 0.1, 5

# SMOOTHNESS_LOWER, SMOOTHNESS_UPPER = 1e-4, 1e-2
LENGTH_SCALE_2_LOWER, LENGTH_SCALE_2_UPPER = 0.1, 10
PERIOD_2_LOWER, PERIOD_2_UPPER = 0.1, 10
A_LOWER, A_UPPER = 0, 0.9

BOUNDS_WN = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
]

BOUNDS_GP = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
    (PERIOD_LOWER, PERIOD_UPPER),
]

BOUNDS_GPC = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
    (PERIOD_LOWER, PERIOD_UPPER),
    (LENGTH_SCALE_2_LOWER, LENGTH_SCALE_2_UPPER),
    (PERIOD_2_LOWER, PERIOD_2_UPPER),
    (A_LOWER, A_UPPER),
]

INITIAL_PARAMS_WN = [0.291450707195285]
INITIAL_PARAMS_GP = [5, 0.3]
INITIAL_PARAMS_GPC = [5, 0.3, 1, 0.3, 0.5]

HYPERPARAM_RULE_DICT_WN = {
    "sigma_max": "multiply",
}

HYPERPARAM_RULE_DICT_GP = {
    "sigma_max": "multiply",
    "period": "multiply",
}

HYPERPARAM_RULE_DICT_GPC = {
    "sigma_max": "multiply",
    "period": "multiply",
    "length_scale_2": "multiply",
    "period_2": "multiply",
    "a": "replace",
}

dts = [5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]

ID = "0001"
R = bgp.get_residual_data(big=True, data_type="news")
param_dict = bgp.get_param_dict(data_type="news")
sim_main = bgp.SXS_CCE(ID, type="news", lev="Lev5", radius="R2") 

R_dict_GP = {} 
analysis_times = np.arange(RESIDUAL_BIG_START, RESIDUAL_BIG_START + RESIDUAL_BIG_END, TIME_STEP)
mask_GP = (analysis_times >= TRAINING_START_TIME_GP) & (analysis_times < TRAINING_START_TIME_GP + TRAINING_RANGE_GP) 

data_times = analysis_times[mask_GP]
for id in R.keys():
    R_dict_GP[id] = {key: value[mask_GP] for key, value in R[id].items()}

def get_hyperparameters_dt(training_spherical_modes=TRAINING_SPH_MODES):
    hyperparameters_array_dt = np.zeros((len(dts), len(INITIAL_PARAMS_GP)))

    initial_params = INITIAL_PARAMS_GP

    for i, dt in enumerate(dts):

        print(f"dt = {dt}")

        new_times = np.arange(TRAINING_START_TIME_GP, TRAINING_START_TIME_GP + TRAINING_RANGE_GP, dt)
        R_dict_interp = {k: bgp.sim_interpolator_data(v, data_times, new_times) for k, v in R_dict_GP.items()}

        hyperparam_list, le, tuned_params = bgp.train_hyper_params(
            TRAINING_START_TIME_GP,
            TRAINING_RANGE_GP,
            dt,
            initial_params,
            BOUNDS_GP,
            param_dict,
            R_dict_interp,
            HYPERPARAM_RULE_DICT_GP,
            bgp.kernel_GP,
            training_spherical_modes,
            SIM_TRAINING_MODE_RULES,
        )

        hyperparameters_array_dt[i, :] = hyperparam_list

        initial_params = hyperparam_list

    return hyperparameters_array_dt


def plot_hyperparameters_dt(hyperparameters_array_dt):
    custom_colormap = LinearSegmentedColormap.from_list("custom_colormap2", config.colors2)
    colors = custom_colormap(np.linspace(0, 1, len(INITIAL_PARAMS_GP)))

    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))

    labels = [r"$\lambda$", r"$\nu$"]

    for i in range(len(INITIAL_PARAMS_GP)):
        ax.plot(dts, hyperparameters_array_dt[:, i], label=labels[i], color=colors[i])

    ax.set_ylabel("Pooled parameter")
    # ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t \, [M]$")
    ax.set_xlim(0, 1)
    #ax.set_ylim(0, 3)
    ax.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    fig.savefig("outputs/hyperparameters_dt.pdf", dpi=600, bbox_inches="tight")


def plot_parameters_dt():

    param_indices = [-1, -2, 0]
    quantiles = [0.05, 0.5, 0.95]

    custom_colormap = LinearSegmentedColormap.from_list("custom_colormap2", config.colors2)
    colors = custom_colormap(np.linspace(0, 1, len(param_indices)))

    fig, (ax_wn, ax_gp) = plt.subplots(2, 1, sharex=True, figsize=(config.fig_width, config.fig_height * 1.5))

    widths_GP = np.zeros((len(dts), len(param_indices)))
    widths_WN = np.zeros((len(dts), len(param_indices)))

    N_MAX = 6
    T0_REF = 17
    T = 100
    
    tuned_param_dict_GP = bgp.get_tuned_param_dict(kernel_type="GP", data_type="news")[ID]
    tuned_param_dict_WN = bgp.get_tuned_param_dict(kernel_type="WN", data_type="news")[ID]
    qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)]
    spherical_modes = [(2, 2)]

    for i, dt in enumerate(dts):

        print(f"dt = {dt}")

        sim_times_interp = np.arange(sim_main.times[0], sim_main.times[-1] + dt, dt)
        sim_h_interp = bgp.sim_interpolator_data(sim_main.h, sim_main.times, sim_times_interp)

        fit_GP = bgp.BGP_fit(
            sim_times_interp,
            sim_h_interp,
            qnm_list,
            sim_main.Mf,
            sim_main.chif_mag,
            tuned_param_dict_GP,
            bgp.kernel_GP,
            t0=T0_REF,
            num_samples=int(1e6),
            t0_method="geq",
            T=T,
            spherical_modes=spherical_modes,
            include_chif=True,
            include_Mf=True,
            data_type="news",
        )

        fit_WN = bgp.BGP_fit(
            sim_times_interp,
            sim_h_interp,
            qnm_list,
            sim_main.Mf,
            sim_main.chif_mag,
            tuned_param_dict_WN,
            bgp.kernel_WN,
            t0=T0_REF,
            num_samples=int(1e6),
            t0_method="geq",
            T=T,
            spherical_modes=spherical_modes,
            include_chif=True,
            include_Mf=True,
            data_type="news",
        )

        samples_GP = fit_GP.fit["samples"]
        samples_WN = fit_WN.fit["samples"]

        for j, param_index in enumerate(param_indices):
            param_samples_GP = samples_GP[:, param_index]
            param_samples_WN = samples_WN[:, param_index]
            quantile_values_GP = np.quantile(param_samples_GP, quantiles)
            quantile_values_WN = np.quantile(param_samples_WN, quantiles)
            widths_GP[i, j] = quantile_values_GP[2] - quantile_values_GP[0]
            widths_WN[i, j] = quantile_values_WN[2] - quantile_values_WN[0]

    param_labels = [r"$M$", r"$\chi$", r"Re($C_{(2,2,0,+)}$)"]

    for j in range(len(param_indices)):
        ax_wn.plot(dts, widths_WN[:, j], color=colors[j], linestyle="--", label=f"{param_labels[j]}")
        ax_gp.plot(dts, widths_GP[:, j], color=colors[j], label=param_labels[j])

    ax_wn.set_ylabel(r"$90 \%$ width (WN)")
    ax_wn.set_xlim(0, 1)
    ax_wn.set_ylim(0, np.max(widths_WN[4:, :]) * 1.1)
    ax_wn.legend(loc="lower right", ncol=2)

    ax_gp.set_ylabel(r"$90 \%$ width (GP)")
    ax_gp.set_xlim(0, 1)
    ax_gp.set_ylim(0, np.max(widths_GP[4:, :]) * 1.1)
    ax_gp.legend(loc="lower right", ncol=2)

    ax_gp.sharex(ax_wn)
    ax_gp.set_xlabel(r"$\Delta t \, [M]$")

    plt.tight_layout()
    fig.savefig("outputs/parameters_dt.pdf", dpi=600, bbox_inches="tight")


if __name__ == "__main__":

    #hyperparameters_array_dt = get_hyperparameters_dt()

    #with open("hyperparameters_array_dt.pkl", "wb") as f:
    #    pickle.dump(hyperparameters_array_dt, f)

    # for mode in [(2,2), (4,4)]:
    #    hyperparameters_array_dt = get_hyperparameters_dt(training_spherical_modes=[mode])
    #    with open(f"hyperparameters_array_dt_{mode}.pkl", "wb") as f:
    #        pickle.dump(hyperparameters_array_dt, f)

    #with open("hyperparameters_array_dt.pkl", "rb") as f:
    #    hyperparameters_array_dt = pickle.load(f)

    #plot_hyperparameters_dt(hyperparameters_array_dt)
    plot_parameters_dt()
