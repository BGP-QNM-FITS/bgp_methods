import numpy as np
import pickle
import matplotlib.pyplot as plt
from CCE import * 
import bgp_qnm_fits as bgp
from matplotlib.colors import LinearSegmentedColormap
from plot_config import PlotConfig
from matplotlib.lines import Line2D

config = PlotConfig()
config.apply_style()

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

SMOOTHNESS = 16
EPSILON = 1 / 10

# These determine the parameter and training range but do not have to match `analysis times' used later.

TRAINING_START_TIME = -10
TRAINING_END_TIME = 100
TIME_STEP = 0.1

analysis_times = np.arange(
    TRAINING_START_TIME,
    TRAINING_START_TIME + TRAINING_END_TIME,
    TIME_STEP,
)  

# Define training bounds

SIGMA_MAX_LOWER, SIGMA_MAX_UPPER = 0.05, 5
T_S_LOWER, T_S_UPPER = -20, 30
LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER = 0.1, 5
PERIOD_LOWER, PERIOD_UPPER = 0.1, 5

SMOOTHNESS_LOWER, SMOOTHNESS_UPPER = 0, 30
LENGTH_SCALE_2_LOWER, LENGTH_SCALE_2_UPPER = 0.1, 5
PERIOD_2_LOWER, PERIOD_2_UPPER = 0.1, 5
A_LOWER, A_UPPER = 0, 0.9

BOUNDS_WN = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
]

BOUNDS_GP = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
    (T_S_LOWER, T_S_UPPER),
    (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER),
    (PERIOD_LOWER, PERIOD_UPPER),
]

BOUNDS_GPC = [
    (SIGMA_MAX_LOWER, SIGMA_MAX_UPPER),
    (T_S_LOWER, T_S_UPPER),
    (SMOOTHNESS_LOWER, SMOOTHNESS_UPPER),
    (LENGTH_SCALE_LOWER, LENGTH_SCALE_UPPER),
    (PERIOD_LOWER, PERIOD_UPPER),
    (LENGTH_SCALE_2_LOWER, LENGTH_SCALE_2_UPPER),
    (PERIOD_2_LOWER, PERIOD_2_UPPER),
    (A_LOWER, A_UPPER),
]

INITIAL_PARAMS_WN = [0.29127733345656215]
INITIAL_PARAMS_GP = [
    0.2283378440307793,
    18.37394010821784,
    0.8610899535603144,
    0.2605530172829033,
]
INITIAL_PARAMS_GPC = [
    0.29443340366568055,
    17.00880319694479,
    8.52342433839235,
    0.992215662729599,
    0.29792754163345136,
    1.4599452640915012,
    3.702622948813973,
    0.8844594560538273,
]

HYPERPARAM_RULE_DICT_WN = {
    "sigma_max": "multiply",
}

HYPERPARAM_RULE_DICT_GP = {
    "sigma_max": "multiply",
    "t_s": "sum",
    "length_scale": "multiply",
    "period": "multiply",
}

HYPERPARAM_RULE_DICT_GPC = {
    "sigma_max": "multiply",
    "t_s": "sum",
    "smoothness": "replace",
    "length_scale": "multiply",
    "period": "multiply",
    "length_scale_2": "multiply",
    "period_2": "multiply",
    "a": "replace",
}

dts = [5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]

ID = "0001"
R = bgp.get_residual_data()
param_dict = bgp.get_param_dict()
sim_main = SXS_CCE(ID, lev="Lev5", radius="R2")

def get_hyperparameters_dt(training_spherical_modes=TRAINING_SPH_MODES):
    hyperparameters_array_dt = np.zeros((len(dts), len(INITIAL_PARAMS_GP)))

    initial_params = INITIAL_PARAMS_GP

    for i, dt in enumerate(dts):

        print(f"dt = {dt}")

        new_times = np.arange(TRAINING_START_TIME, TRAINING_START_TIME+TRAINING_END_TIME, dt)
        R_dict_interp = {k:bgp.sim_interpolator_data(v, analysis_times, new_times) for k, v in R.items()}

        hyperparam_list, le, tuned_params = bgp.train_hyper_params(
            TRAINING_START_TIME,
            TRAINING_END_TIME,
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(config.fig_width, config.fig_height * 1.5), sharex=True, 
                                  gridspec_kw={'height_ratios': [1, 2]})
    
    labels = [r"$\lambda$", r"$\delta$", r"$\nu$", r"$\mu$"]

    ax1.plot(dts, hyperparameters_array_dt[:, 1], color=colors[1], label=labels[1])
    ax1.set_ylabel("Value")
    ax1.legend()

    for i in [0, 2, 3]:
        ax2.plot(dts, hyperparameters_array_dt[:, i], label=labels[i], color=colors[i])
    ax2.set_ylabel("Value")
    ax2.set_xscale("log")
    ax2.set_xlabel("dt")
    ax2.legend(loc = "upper right", ncol=3)

    ax1.set_xlim(dts[-1], dts[0])
    ax2.set_xlim(dts[-1], dts[0])

    # Reduce space between subplots
    plt.subplots_adjust(hspace=0.02)
    
    # Make sure x-axis ticks appear only on the bottom plot
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    plt.tight_layout()
    fig.savefig("outputs/hyperparameters_dt.pdf", dpi=600, bbox_inches="tight")


def plot_parameters_dt():

    N_MAX = 6
    T0_REF = 17
    T = 100 
    tuned_param_dict_GP = bgp.get_param_data("GP")[ID]
    qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)]
    spherical_modes = [(2, 2)]

    fits = [] 

    for i, dt in enumerate(dts):
        sim_times_interp = np.arange(sim_main.times[0], sim_main.times[0]+sim_main.times[-1], dt)
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
            )

    fits.append(fit_GP)


if __name__ == "__main__":

    hyperparameters_array_dt = get_hyperparameters_dt()

    with open("hyperparameters_array_dt.pkl", "wb") as f:
        pickle.dump(hyperparameters_array_dt, f)

    for mode in [(2,2), (4,4)]:
        hyperparameters_array_dt = get_hyperparameters_dt(training_spherical_modes=[mode])
        with open(f"hyperparameters_array_dt_{mode}.pkl", "wb") as f:
            pickle.dump(hyperparameters_array_dt, f)

    with open("hyperparameters_array_dt.pkl", "rb") as f:
        hyperparameters_array_dt = pickle.load(f)

    plot_hyperparameters_dt(hyperparameters_array_dt)