import qnmfits
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import HTMLWriter
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from CCE import SXS_CCE
import bgp_qnm_fits as bgp

from scipy.optimize import minimize

from plot_config import PlotConfig
import corner
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
from matplotlib.animation import PillowWriter


class MethodPlots:

    config = PlotConfig()
    config.apply_style()

    def __init__(
        self,
        id,
        N_MAX=7,
        T=100,
        T0_REF=17,
        num_samples=10000,
        include_Mf=True,
        include_chif=True,
        use_nonlinear_params=False,
        decay_corrected=False,
    ):
        self.id = id
        self.N_MAX = N_MAX
        self.T = T
        self.T0_REF = T0_REF
        self.include_Mf = include_Mf
        self.include_chif = include_chif
        self.num_samples = num_samples
        self.use_nonlinear_params = use_nonlinear_params
        self.decay_corrected = decay_corrected

        self.sim_main = SXS_CCE(id, lev="Lev5", radius="R2")
        self.sim_lower = SXS_CCE(id, lev="Lev4", radius="R2")

        self._align_waveforms()

        self.qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)]
        self.spherical_modes = [(2, 2)]

        # Get the true values for the spin and mass (i.e. the bondi data from metadata file)

        self.chif_mag_ref = self.sim_main.chif_mag
        self.Mf_ref = self.sim_main.Mf

        self.T0s = np.linspace(-10, 100, 110)

        self._initialize_results()

        colors = self.config.colors
        self.custom_colormap = LinearSegmentedColormap.from_list("custom_colormap", colors)
        self.fundamental_color_WN = to_hex("#395470")
        self.fundamental_color_GP = to_hex("#395471")
        self.overtone_color_WN = to_hex("#65858c")
        self.overtone_color_GP = to_hex("#65858d")
        self.extra_mode_color = to_hex("#DE6A5E")

    def _align_waveforms(self):
        """
        Align waveforms and interpolate them onto the same time grid.
        """
        time_shift = bgp.get_time_shift(self.sim_main, self.sim_lower)
        self.sim_lower.zero_time = -time_shift
        self.sim_lower.time_shift()

        new_times = np.arange(self.sim_main.times[0], self.sim_main.times[-1], 0.1)
        self.sim_main_interp = bgp.sim_interpolator(self.sim_main, new_times)
        self.sim_lower_interp = bgp.sim_interpolator(self.sim_lower, new_times)

    def _initialize_results(self):
        """
        Initialize arrays to store results for mismatches, amplitudes, and significance.
        """
        num_T0s = len(self.T0s)
        num_modes = len(self.qnm_list)

        self.unweighted_mismatches_LS = np.zeros(num_T0s)
        self.weighted_mismatches_LS = np.zeros(num_T0s)

        self.unweighted_mismatches_WN = np.zeros(num_T0s)
        self.weighted_mismatches_WN = np.zeros(num_T0s)

        self.unweighted_mismatches_GP = np.zeros(num_T0s)
        self.weighted_mismatches_GP = np.zeros(num_T0s)

        self.unweighted_mismatches_noise = np.zeros(num_T0s)
        self.weighted_mismatches_noise = np.zeros(num_T0s)

        self.log_likelihood_WN = np.zeros(num_T0s)
        self.log_likelihood_GP = np.zeros(num_T0s)

        self.amplitudes_LS = np.zeros((num_T0s, num_modes))

        self.amplitudes_WN = np.zeros((num_T0s, num_modes))
        self.amplitudes_GP = np.zeros((num_T0s, num_modes))

        self.amplitudes_WN_percentiles = {
            p: np.zeros((num_T0s, num_modes)) for p in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        }
        self.amplitudes_GP_percentiles = {
            p: np.zeros((num_T0s, num_modes)) for p in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        }

        self.significances_WN = np.zeros((num_T0s, num_modes))
        self.significances_GP = np.zeros((num_T0s, num_modes))

        self.neffs_WN = np.zeros(num_T0s)
        self.neffs_GP = np.zeros(num_T0s)

        self.linear_approx_mismatch_WN = np.zeros(num_T0s)
        self.linear_approx_mismatch_GP = np.zeros(num_T0s)

        self.samples_WN = np.zeros((num_T0s, self.num_samples, 2 * len(self.qnm_list) + 2))
        self.samples_GP = np.zeros((num_T0s, self.num_samples, 2 * len(self.qnm_list) + 2))

        self.mean_vector_WN = np.zeros((num_T0s, 2 * len(self.qnm_list) + 2))
        self.mean_vector_GP = np.zeros((num_T0s, 2 * len(self.qnm_list) + 2))

    def load_tuned_parameters(self):
        """
        Load tuned kernel parameters for GP and WN fits.
        """
        self.tuned_param_dict_GP = bgp.get_param_data("GP")[self.id]
        self.tuned_param_dict_WN = bgp.get_param_data("WN")[self.id]
        self.tuned_param_dict_GPC = bgp.get_param_data("GPc")[self.id]

    def compute_mf_chif(self):
        """
        Compute mass and spin parameters for each t0 using least-squares minimization.
        """
        initial_params = (self.Mf_ref, self.chif_mag_ref)
        Mf_RANGE = (self.Mf_ref * 0.5, self.Mf_ref * 1.5)
        chif_mag_RANGE = (0.1, 0.99)
        bounds = (Mf_RANGE, chif_mag_RANGE)

        # Get T0 reference values first (i.e. the values at t0 = 17)

        result = minimize(
            self._mf_chif_mismatch,
            initial_params,
            args=(self.T0_REF, self.T, self.spherical_modes),
            method="Nelder-Mead",
            bounds=bounds,
        )

        self.Mf_t0 = result.x[0]
        self.chif_t0 = result.x[1]

        self.Mfs_chifs = np.zeros((len(self.T0s), 2))

        for i, t0 in enumerate(self.T0s):
            args = (t0, self.T, self.spherical_modes)
            result = minimize(
                self._mf_chif_mismatch,
                initial_params,
                args=args,
                method="Nelder-Mead",
                bounds=bounds,
            )
            self.Mfs_chifs[i] = result.x
            initial_params = result.x

    def _mf_chif_mismatch(self, Mf_chif_mag_list, t0, T, spherical_modes):
        """
        Compute the mismatch for given mass and spin parameters.
        """
        Mf, chif_mag = Mf_chif_mag_list
        best_fit = qnmfits.multimode_ringdown_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            Mf,
            chif_mag,
            t0,
            t0_method="closest",
            T=T,
            spherical_modes=spherical_modes,
        )
        return best_fit["mismatch"]

    def _compute_LS_fits(self):
        """
        Compute GP, WN, and LS fits for each t0 and store results.
        """

        fits_LS = []
        main_data_masked = []
        lower_data_masked = []

        for i, t0 in enumerate(self.T0s):

            Mf = self.Mf_ref
            chif_mag = self.chif_mag_ref

            if self.include_Mf:
                Mf = self.Mfs_chifs[i, 0]
            if self.include_chif:
                chif_mag = self.Mfs_chifs[i, 1]

            fit_LS = qnmfits.multimode_ringdown_fit(
                self.sim_main.times,
                self.sim_main.h,
                self.qnm_list,
                Mf,
                chif_mag,
                t0,
                T=self.T,
                spherical_modes=self.spherical_modes,
            )

            fits_LS.append(fit_LS)

            mm_mask = (self.sim_main_interp.times >= t0 - 1e-9) & (self.sim_main_interp.times < t0 + self.T - 1e-9)
            main_data = np.array([self.sim_main_interp.h[(2, 2)][mm_mask]])
            lower_data = np.array([self.sim_lower_interp.h[(2, 2)][mm_mask]])

            main_data_masked.append(main_data)
            lower_data_masked.append(lower_data)

        return fits_LS, main_data_masked, lower_data_masked

    def compute_quantities(self):

        fits_LS, main_data_masked, lower_data_masked = self._compute_LS_fits()

        fits_WN = bgp.BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.tuned_param_dict_WN,
            bgp.kernel_WN,
            t0=self.T0s,
            use_nonlinear_params=self.use_nonlinear_params,
            decay_corrected=self.decay_corrected,
            num_samples=self.num_samples,
            t0_method="closest",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
        )

        fits_GP = bgp.BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.tuned_param_dict_GP,
            bgp.kernel_GP,
            t0=self.T0s,
            use_nonlinear_params=self.use_nonlinear_params,
            decay_corrected=self.decay_corrected,
            num_samples=self.num_samples,
            t0_method="closest",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
        )

        self._store_results(fits_GP.fits, fits_WN.fits, fits_LS, main_data_masked, lower_data_masked)

    def _store_results(self, fit_GP, fit_WN, fit_LS, main_data, lower_data):
        """
        Store results for mismatches, amplitudes, and significance for a given t0.
        """

        for i, t0 in enumerate(self.T0s):

            # Mismatches

            model_array_LS = np.array([fit_LS[i]["model"][key] for key in fit_LS[i]["model"].keys()])
            data_array_LS = np.array([fit_LS[i]["data"][key] for key in fit_LS[i]["data"].keys()])

            self.unweighted_mismatches_LS[i] = bgp.mismatch(model_array_LS, data_array_LS)
            self.weighted_mismatches_LS[i] = bgp.mismatch(
                model_array_LS, data_array_LS, fit_GP[i]["inv_noise_covariance"]
            )

            self.unweighted_mismatches_WN[i] = bgp.mismatch(
                fit_WN[i]["model_array_linear"], fit_WN[i]["data_array_masked"]
            )
            self.weighted_mismatches_WN[i] = bgp.mismatch(
                fit_WN[i]["model_array_linear"],
                fit_WN[i]["data_array_masked"],
                fit_GP[i]["inv_noise_covariance"],
            )

            self.unweighted_mismatches_GP[i] = bgp.mismatch(
                fit_GP[i]["model_array_linear"], fit_GP[i]["data_array_masked"]
            )
            self.weighted_mismatches_GP[i] = bgp.mismatch(
                fit_GP[i]["model_array_linear"],
                fit_GP[i]["data_array_masked"],
                fit_GP[i]["inv_noise_covariance"],
            )

            self.unweighted_mismatches_noise[i] = bgp.mismatch(main_data[i], lower_data[i])
            self.weighted_mismatches_noise[i] = bgp.mismatch(
                main_data[i], lower_data[i], fit_GP[i]["inv_noise_covariance"]
            )

            # Log likelihood

            self.log_likelihood_WN[i] = -bgp.log_likelihood(
                fit_WN[i]["data_array_masked"],
                fit_WN[i]["model_array_linear"],
                fit_GP[i]["inv_noise_covariance"],
            )
            self.log_likelihood_GP[i] = -bgp.log_likelihood(
                fit_GP[i]["data_array_masked"],
                fit_GP[i]["model_array_linear"],
                fit_GP[i]["inv_noise_covariance"],
            )

            # Amplitudes
            self.amplitudes_LS[i, :] = np.abs(fit_LS[i]["C"])
            self.amplitudes_WN[i, :] = fit_WN[i]["mean_amplitude"]
            self.amplitudes_GP[i, :] = fit_GP[i]["mean_amplitude"]

            for p in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
                self.amplitudes_WN_percentiles[p][i, :] = fit_WN[i]["unweighted_quantiles"][p]
                self.amplitudes_GP_percentiles[p][i, :] = fit_GP[i]["unweighted_quantiles"][p]

            # Significance
            self.significances_WN[i, :] = bgp.get_significance_list(
                self.qnm_list,
                np.array(fit_WN[i]["mean"]),
                np.array(fit_WN[i]["fisher_matrix"]),
                include_chif=self.include_chif,
                include_Mf=self.include_Mf,
            )
            self.significances_GP[i, :] = bgp.get_significance_list(
                self.qnm_list,
                np.array(fit_GP[i]["mean"]),
                np.array(fit_GP[i]["fisher_matrix"]),
                include_chif=self.include_chif,
                include_Mf=self.include_Mf,
            )

            # Samples
            self.samples_WN[i, :, :] = fit_WN[i]["samples"]
            self.samples_GP[i, :, :] = fit_GP[i]["samples"]

            self.mean_vector_WN[i, :] = fit_WN[i]["mean"]
            self.mean_vector_GP[i, :] = fit_GP[i]["mean"]

            self.neffs_WN[i] = np.sum(fit_WN[i]["samples_weights"]) ** 2 / np.sum(fit_WN[i]["samples_weights"] ** 2)
            self.neffs_GP[i] = np.sum(fit_GP[i]["samples_weights"]) ** 2 / np.sum(fit_GP[i]["samples_weights"] ** 2)

    def plot_mismatch(self, output_path="outputs/mismatch_plot.pdf", show=False):
        """
        Generate the mismatch plot and save it to the specified path.
        """
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(self.config.fig_width, self.config.fig_height * 1.7),
            sharex=True,
            gridspec_kw={"hspace": 0},
        )

        # Plot weighted mismatches
        ax1.axvline(self.T0_REF, color="k", alpha=0.3, lw=1)
        ax1.plot(self.T0s, self.weighted_mismatches_GP, label="GP", color="k")
        ax1.plot(self.T0s, self.weighted_mismatches_WN, label="WN", ls="--", color="k")
        ax1.fill_between(self.T0s, 0, self.weighted_mismatches_noise, color="grey", alpha=0.5)
        ax1.set_xlim(self.T0s[0], self.T0s[-1])
        ax1.set_ylabel(r"$\mathcal{M}_{\rm GP}$")
        ax1.set_yscale("log")
        ax1.legend(frameon=False, loc="upper right", labelspacing=0.1)

        # Plot unweighted mismatches
        ax2.axvline(self.T0_REF, color="k", alpha=0.3, lw=1)
        ax2.plot(self.T0s, self.unweighted_mismatches_GP, label="GP", color="k")
        ax2.plot(self.T0s, self.unweighted_mismatches_WN, label="WN", ls="--", color="k")
        ax2.fill_between(self.T0s, 0, self.unweighted_mismatches_noise, color="grey", alpha=0.5)
        ax2.set_xlim(self.T0s[0], self.T0s[-1])
        ax2.set_xlabel("$t_0 [M]$")
        ax2.set_ylabel(r"$\mathcal{M}_{\rm WN}$")
        ax2.set_yscale("log")
        ax2.legend(frameon=False, loc="upper right", labelspacing=0.1)

        # Save and/or show the plot
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_log_likelihood(self, output_path="outputs/log_likelihood_mismatch_plot.pdf", show=False):
        """
        Generate the mismatch plot and save it to the specified path.
        """
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(self.config.fig_width, self.config.fig_height * 1.7),
            sharex=True,
            gridspec_kw={"hspace": 0},
        )

        # Plot weighted mismatches
        ax1.axvline(self.T0_REF, color="k", alpha=0.3, lw=1)
        ax1.plot(self.T0s, self.log_likelihood_GP, label="GP", color="k")
        ax1.plot(self.T0s, self.log_likelihood_WN, label="WN", ls="--", color="k")
        ax1.fill_between(self.T0s, 0, self.weighted_mismatches_noise, color="grey", alpha=0.5)
        ax1.set_xlim(self.T0s[0], self.T0s[-1])
        ax1.set_ylabel(r"$-\ln P(\mathfrak{h}|\theta)$")
        ax1.set_yscale("log")
        ax1.legend(frameon=False, loc="upper right", labelspacing=0.1)

        # Plot unweighted mismatches
        ax2.axvline(self.T0_REF, color="k", alpha=0.3, lw=1)
        ax2.plot(self.T0s, self.unweighted_mismatches_GP, label="GP", color="k")
        ax2.plot(self.T0s, self.unweighted_mismatches_WN, label="WN", ls="--", color="k")
        ax2.fill_between(self.T0s, 0, self.unweighted_mismatches_noise, color="grey", alpha=0.5)
        ax2.set_xlim(self.T0s[0], self.T0s[-1])
        ax2.set_xlabel("$t_0 [M]$")
        ax2.set_ylabel(r"$\mathcal{M}^{22}$")
        ax2.set_yscale("log")
        ax2.legend(frameon=False, loc="upper right", labelspacing=0.1)

        # Save and/or show the plot
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_amplitude(self, output_path="outputs/amplitude_plot.pdf", show=False):
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))
        colors = self.custom_colormap(np.linspace(0, 1, len(self.qnm_list)))

        for i, qnm in enumerate(self.qnm_list):
            # decay_time = qnmfits.qnm.omega_list([qnm], self.chif_mag_ref, self.Mf_ref)[0].imag
            # closest_time_index = np.argmin(np.abs(self.T0s - 0))
            # C_tau = np.exp(decay_time * (self.T0s[closest_time_index] - self.T0s))
            ax.plot(
                self.T0s,
                self.amplitudes_GP_percentiles[0.5][:, i],
                label=f"{qnm[2]}",
                color=colors[i],
            )
            ax.plot(
                self.T0s,
                self.amplitudes_WN_percentiles[0.5][:, i],
                linestyle="--",
                color=colors[i],
            )
            ax.fill_between(
                self.T0s,
                self.amplitudes_GP_percentiles[0.1][:, i],
                self.amplitudes_GP_percentiles[0.9][:, i],
                alpha=0.2,
                color=colors[i],
            )
            ax.fill_between(
                self.T0s,
                self.amplitudes_WN_percentiles[0.1][:, i],
                self.amplitudes_WN_percentiles[0.9][:, i],
                alpha=0.2,
                color=colors[i],
            )

        solid_line = Line2D([0], [0], color="black", linestyle="-")
        dashed_line = Line2D([0], [0], color="black", linestyle="--")
        color_legend = ax.legend(
            title="$n$",
            title_fontsize=8,
            ncol=1,
            frameon=False,
            loc="center right",
            bbox_to_anchor=(1.20, 0.5),
            fontsize=7,
        )
        ax.legend(
            [solid_line, dashed_line],
            ["GP", "WN"],
            frameon=False,
            loc="upper left",
            ncol=1,
            fontsize=7,
        )
        ax.add_artist(color_legend)

        ax.axvline(self.T0_REF, color="k", alpha=0.3, lw=1)
        ax.set_xlim(-10, 30)
        ax.set_ylim(1e-1, 1e5)
        ax.set_xlabel("$t_0 [M]$")
        ax.set_ylabel(r"$|\hat{C}_{\alpha}|$")
        ax.set_yscale("log")

        plt.tight_layout()
        plt.subplots_adjust(right=1)
        fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=[color_legend])
        if show:
            plt.show()
        plt.close(fig)

    def plot_significance(self, output_path="outputs/significance_plot.pdf", show=False):
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))

        handles = []
        labels = []

        for i, qnm in enumerate(self.qnm_list):

            if qnm == (3, 2, 0, 1):
                colors = self.custom_colormap(np.linspace(0, 1, len(self.qnm_list) - 1))
                color = self.extra_mode_color
                label = r"$\alpha$"
                lw = 2
            else:
                colors = self.custom_colormap(np.linspace(0, 1, len(self.qnm_list)))
                color = colors[i]
                label = f"{qnm[2]}"
                # label=f"{qnm[2]}"
                lw = 1

            line = ax.plot(
                self.T0s,
                self.significances_GP[:, i],
                label=label,
                color=color,
                lw=lw,
            )[0]
            ax.plot(
                self.T0s,
                self.significances_WN[:, i],
                linestyle="--",
                color=color,
                lw=lw,
            )

            handles.append(line)
            labels.append(label)

        solid_line = Line2D([0], [0], color="black", linestyle="-")
        dashed_line = Line2D([0], [0], color="black", linestyle="--")

        # Create a separate axis for the color legend
        color_legend_ax = fig.add_axes([1.08, 0.45, 0.1, 0.3])  # Adjust position and size as needed
        color_legend_ax.axis("off")  # Turn off the axis

        if (3, 2, 0, 1) in self.qnm_list:
            # Create a smaller spacer with minimal height
            spacer = Line2D([0], [0], color="none", marker=" ", linestyle="", markersize=1)

            # Insert spacer before the alpha mode in the legend
            spacer_idx = labels.index(r"$\alpha$")
            handles.insert(spacer_idx, spacer)
            labels.insert(spacer_idx, " ")

        color_legend = color_legend_ax.legend(
            handles=handles,
            labels=labels,
            title="$n$",
            title_fontsize=8,
            ncol=1,
            frameon=False,
            loc="center",
            fontsize=7,
        )
        fig.add_artist(color_legend)

        ax.legend(
            [solid_line, dashed_line],
            ["GP", "WN"],
            frameon=False,
            loc="upper right",
            ncol=1,
            fontsize=7,
        )

        # ax.set_xticks(self.T0s[::5])
        # ax.set_xticklabels([f"{t0:.1f}" for t0 in self.T0s[::5]], rotation=90, fontsize=6)
        # ax.grid(axis="x", linestyle="-", color="grey", alpha=0.7)

        ax.axvline(self.T0_REF, color="k", alpha=0.3)
        ax.set_xlabel("$t_0 [M]$")
        ax.set_ylabel(r"$\mathcal{S}_{\alpha}$")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-10, 100)

        plt.tight_layout()
        plt.subplots_adjust(right=1)
        fig.savefig(output_path)
        if show:
            plt.show()
        plt.close(fig)

    def plot_spirals_static(self, output_path="outputs/posterior_spiral_static.pdf", show=False):

        # samples_WN = self.samples_WN[i, :, index1:index2+1]
        # samples_GP = self.samples_GP[i, :, index1:index2+1]

        # What is the width of two columns?

        fig = plt.figure(figsize=(self.config.fig_width_2, self.config.fig_height_2))
        grid = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.4, wspace=0.25)
        ax_top = fig.add_subplot(grid[0, :])
        ax_bottom_left = fig.add_subplot(grid[1, 0])
        ax_bottom_center = fig.add_subplot(grid[1, 1])
        ax_bottom_right = fig.add_subplot(grid[1, 2])

        colors = self.custom_colormap(np.linspace(0, 1, len(self.qnm_list)))

        # specified_T0s_1 = [12, 17, 20]
        # specified_T0s_2 = [6, 10, 15]
        # specified_T0s_3 = [5, 8, 12]

        specified_T0s_1 = [13.25, 15.7, 19]
        specified_T0s_2 = [9, 10.5, 13]
        specified_T0s_3 = [9.2, 16.3, 20.3]

        markers = ["o", "^", "s"]

        for i, qnm in enumerate(self.qnm_list):

            # Apply cubic spline to smooth the data on the finer grid
            fine_grid = np.linspace(self.T0s[0], self.T0s[-1], len(self.T0s) * 100)
            smoothed_x = CubicSpline(self.T0s, self.mean_vector_GP[:, 2 * i])(fine_grid)
            smoothed_y = CubicSpline(self.T0s, self.mean_vector_GP[:, 2 * i + 1])(fine_grid)
            smoothed_significance = CubicSpline(self.T0s, self.significances_GP[:, i])(fine_grid)

            if i == 0 or i > 3:
                continue

            ax_top.plot(
                fine_grid,
                smoothed_significance,
                label=f"{qnm[2]}",
                color=colors[i],
            )

            if i == 1:
                ax_bottom_left.plot(
                    smoothed_x,
                    smoothed_y,
                    color=colors[i],
                    linestyle="-",
                    label=f"{qnm[2]}",
                )
                ax_bottom_left.set_xlim(-0.16, 0.16)
                ax_bottom_left.set_ylim(-0.16, 0.16)
                ax_bottom_left.axhline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)
                ax_bottom_left.axvline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)

                for j, t0 in enumerate(specified_T0s_1):
                    idx1 = np.argmin(np.abs(fine_grid - t0))
                    marker = markers[j % 3]
                    ax_top.scatter(
                        fine_grid[idx1],
                        smoothed_significance[idx1],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )
                    ax_bottom_left.scatter(
                        smoothed_x[idx1],
                        smoothed_y[idx1],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )

            elif i == 2:
                ax_bottom_center.plot(
                    smoothed_x,
                    smoothed_y,
                    color=colors[i],
                    linestyle="-",
                    label=f"{qnm[2]}",
                )
                ax_bottom_center.set_xlim(-0.35, 0.35)
                ax_bottom_center.set_ylim(-0.35, 0.35)
                ax_bottom_center.axhline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)
                ax_bottom_center.axvline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)

                for j, t0 in enumerate(specified_T0s_2):
                    idx2 = np.argmin(np.abs(fine_grid - t0))
                    marker = markers[j % 3]
                    ax_top.scatter(
                        fine_grid[idx2],
                        smoothed_significance[idx2],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )
                    ax_bottom_center.scatter(
                        smoothed_x[idx2],
                        smoothed_y[idx2],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )

            elif i == 3:
                ax_bottom_right.plot(
                    smoothed_x,
                    smoothed_y,
                    color=colors[i],
                    linestyle="-",
                    label=f"{qnm[2]}",
                )
                ax_bottom_right.set_xlim(-1.2, 1.2)
                ax_bottom_right.set_ylim(-1.2, 1.2)
                ax_bottom_right.axhline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)
                ax_bottom_right.axvline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)

                for j, t0 in enumerate(specified_T0s_3):
                    idx3 = np.argmin(np.abs(fine_grid - t0))
                    marker = markers[j % 3]
                    ax_top.scatter(
                        fine_grid[idx3],
                        smoothed_significance[idx3],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )
                    ax_bottom_right.scatter(
                        smoothed_x[idx3],
                        smoothed_y[idx3],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )

        ax_top.set_xlim(0, 40)
        ax_top.set_ylim(0.5, 1.05)
        ax_top.set_xlabel("$t_0 [M]$")
        ax_top.set_ylabel(r"$\mathcal{S}_{\alpha}$")
        ax_top.legend(
            title="$n$",
            ncol=1,
            frameon=False,
            loc="lower left",
        )

        ax_bottom_left.set_title(r"$n = 1$")
        ax_bottom_center.set_title(r"$n = 2$")
        ax_bottom_right.set_title(r"$n = 3$")

        ax_bottom_left.set_xlabel(r"$\mathrm{Re}(C_{\alpha})$")
        ax_bottom_left.set_ylabel(r"$\mathrm{Im}(C_{\alpha})$", labelpad=-2)
        ax_bottom_center.set_xlabel(r"$\mathrm{Re}(C_{\alpha})$")
        ax_bottom_right.set_xlabel(r"$\mathrm{Re}(C_{\alpha})$")

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_spirals_static_320(self, output_path="outputs/posterior_spiral_static.pdf", show=False):

        # samples_WN = self.samples_WN[i, :, index1:index2+1]
        # samples_GP = self.samples_GP[i, :, index1:index2+1]

        fig = plt.figure(figsize=(self.config.fig_width_2, self.config.fig_height_2))
        grid = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.4, wspace=0.25)
        ax_top = fig.add_subplot(grid[0, :])
        ax_bottom_left = fig.add_subplot(grid[1, 0])
        ax_bottom_center = fig.add_subplot(grid[1, 1])
        ax_bottom_right = fig.add_subplot(grid[1, 2])

        colors = self.custom_colormap(np.linspace(0, 1, len(self.qnm_list)))

        # specified_T0s_1 = [12, 17, 20]
        # specified_T0s_2 = [6, 10, 15]
        # specified_T0s_3 = [5, 8, 12]

        specified_T0s_1 = [13.25, 15.7, 19]
        specified_T0s_2 = [9, 10.5, 13]
        specified_T0s_3 = [9.2, 16.3, 20.3]

        markers = ["o", "^", "s"]

        for i, qnm in enumerate(self.qnm_list):

            # Apply cubic spline to smooth the data on the finer grid
            fine_grid = np.linspace(self.T0s[0], self.T0s[-1], len(self.T0s) * 100)
            smoothed_x = CubicSpline(self.T0s, self.mean_vector_GP[:, 2 * i])(fine_grid)
            smoothed_y = CubicSpline(self.T0s, self.mean_vector_GP[:, 2 * i + 1])(fine_grid)
            smoothed_significance = CubicSpline(self.T0s, self.significances_GP[:, i])(fine_grid)

            if i == 0 or i > 3:
                continue

            ax_top.plot(
                fine_grid,
                smoothed_significance,
                label=f"{qnm[2]}",
                color=colors[i],
            )

            if i == 1:
                ax_bottom_left.plot(
                    smoothed_x,
                    smoothed_y,
                    color=colors[i],
                    linestyle="-",
                    label=f"{qnm[2]}",
                )
                ax_bottom_left.set_xlim(-0.2, 0.2)
                ax_bottom_left.set_ylim(-0.2, 0.2)
                ax_bottom_left.axhline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)
                ax_bottom_left.axvline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)

                for j, t0 in enumerate(specified_T0s_1):
                    idx1 = np.argmin(np.abs(fine_grid - t0))
                    marker = markers[j % 3]
                    ax_top.scatter(
                        fine_grid[idx1],
                        smoothed_significance[idx1],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )
                    ax_bottom_left.scatter(
                        smoothed_x[idx1],
                        smoothed_y[idx1],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )

            elif i == 2:
                ax_bottom_center.plot(
                    smoothed_x,
                    smoothed_y,
                    color=colors[i],
                    linestyle="-",
                    label=f"{qnm[2]}",
                )
                ax_bottom_center.set_xlim(-0.42, 0.42)
                ax_bottom_center.set_ylim(-0.42, 0.42)
                ax_bottom_center.axhline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)
                ax_bottom_center.axvline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)

                for j, t0 in enumerate(specified_T0s_2):
                    idx2 = np.argmin(np.abs(fine_grid - t0))
                    marker = markers[j % 3]
                    ax_top.scatter(
                        fine_grid[idx2],
                        smoothed_significance[idx2],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )
                    ax_bottom_center.scatter(
                        smoothed_x[idx2],
                        smoothed_y[idx2],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )

            elif i == 3:
                ax_bottom_right.plot(
                    smoothed_x,
                    smoothed_y,
                    color=colors[i],
                    linestyle="-",
                    label=f"{qnm[2]}",
                )
                ax_bottom_right.set_xlim(-1.2, 1.2)
                ax_bottom_right.set_ylim(-1.2, 1.2)
                ax_bottom_right.axhline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)
                ax_bottom_right.axvline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)

                for j, t0 in enumerate(specified_T0s_3):
                    idx3 = np.argmin(np.abs(fine_grid - t0))
                    marker = markers[j % 3]
                    ax_top.scatter(
                        fine_grid[idx3],
                        smoothed_significance[idx3],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )
                    ax_bottom_right.scatter(
                        smoothed_x[idx3],
                        smoothed_y[idx3],
                        color=colors[i],
                        marker=marker,
                        s=7,
                    )

        ax_top.set_xlim(0, 40)
        ax_top.set_ylim(-0.05, 1.05)
        ax_top.set_xlabel("$t_0 [M]$")
        ax_top.set_ylabel(r"$\mathcal{S}_{\alpha}$")
        ax_top.legend(
            title="$n$",
            ncol=1,
            frameon=False,
            loc="lower left",
        )

        ax_bottom_left.set_title(r"$n = 1$")
        ax_bottom_center.set_title(r"$n = 2$")
        ax_bottom_right.set_title(r"$n = 3$")

        ax_bottom_left.set_xlabel(r"$\mathrm{Re}(C_{\alpha})$")
        ax_bottom_left.set_ylabel(r"$\mathrm{Im}(C_{\alpha})$", labelpad=-2)
        ax_bottom_center.set_xlabel(r"$\mathrm{Re}(C_{\alpha})$")
        ax_bottom_right.set_xlabel(r"$\mathrm{Re}(C_{\alpha})$")

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_spirals_ani(self, output_path="outputs/posterior_spiral.pdf", show=False):

        index1 = 2
        index2 = 3

        labels = [
            r"$\mathrm{Re}(C_{(2,2,1,+)})$",
            r"$\mathrm{Im}(C_{(2,2,1,+)})$",
        ]

        fig = corner.corner(
            self.samples_WN[0, :, index1 : index2 + 1],
            color="blue",
            show_titles=True,
            levels=[0.1, 0.5, 0.9],
        )

        def update(i):

            fig.clear()

            samples_WN = self.samples_WN[i, :, index1 : index2 + 1]
            samples_GP = self.samples_GP[i, :, index1 : index2 + 1]

            corner.corner(
                samples_WN,
                color="blue",
                labels=labels,
                fig=fig,
                show_titles=True,
                levels=[0.1, 0.5, 0.9],
            )

            corner.corner(
                samples_GP,
                color="red",
                labels=labels,
                fig=fig,
                show_titles=True,
                levels=[0.1, 0.5, 0.9],
            )

            fig.axes[1].set_visible(True)  # Ensure the axis is visible
            fig.axes[1].tick_params(axis="both", which="both", direction="in", top=True, right=True)
            fig.axes[1].spines["top"].set_visible(True)
            fig.axes[1].spines["right"].set_visible(True)
            fig.axes[1].set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
            fig.axes[1].set_yticks([-0.2, -0.1, 0, 0.1, 0.2])

            fig.axes[1].plot(
                self.mean_vector_WN[: i + 1, index1],
                self.mean_vector_WN[: i + 1, index2],
                "-",
                color="blue",
                alpha=0.7,
                label="Trajectory WN",
            )
            fig.axes[1].plot(
                self.mean_vector_GP[: i + 1, index1],
                self.mean_vector_GP[: i + 1, index2],
                "-",
                color="red",
                alpha=0.7,
                label="Trajectory GP",
            )

            fig.axes[1].axhline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)
            fig.axes[1].axvline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)

            fig.axes[1].set_xlim(-0.2, 0.2)
            fig.axes[1].set_ylim(-0.2, 0.2)

            fig.axes[2].axhline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)
            fig.axes[2].axvline(0, color="black", alpha=0.5, linestyle="--", lw=0.5)

            limits = 0.2

            fig.axes[0].set_xlim(-limits, limits)
            fig.axes[3].set_xlim(-limits, limits)
            fig.axes[2].set_xlim(-limits, limits)
            fig.axes[2].set_ylim(-limits, limits)

            fig.suptitle(f"{self.T0s[i]:.1f} M", fontsize=10)

        ani = FuncAnimation(fig, update, frames=len(self.T0s), repeat=False)
        ani.save("outputs/amplitude_spiral.gif", writer=PillowWriter(fps=10))
        ani.save("outputs/amplitude_spiral.html", writer=HTMLWriter(fps=10))
        if show:
            plt.show()
        plt.close(fig)

    def diagnostic_plots(self):

        plt.plot(self.T0s, self.mismatches_WN, label="WN", lw=2, color="k")

        plt.plot(self.T0s, self.linear_approx_mismatch_WN, label="WN", lw=2, color="k")
        plt.plot(self.T0s, self.linear_approx_mismatch_GP, label="GP", lw=2, color="r")
        plt.xlabel("$t_0 [M]$")
        plt.ylabel(r"$\mathcal{M}$")
        plt.yscale("log")
        plt.legend()
        plt.show()

        plt.plot(self.T0s, self.neffs_WN, label="WN", lw=2, color="k")
        plt.plot(self.T0s, self.neffs_GP, label="GP", lw=2, color="r")
        plt.show()

        plt.plot(self.model_times_WN, self.residual_WN, label="WN", lw=2, color="k")
        plt.plot(self.model_times_GP, self.residual_GP, label="GP", lw=2, color="r")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.title("Residuals of the Fits")
        plt.legend()
        plt.show()

        plt.hist(self.samples_weights_GP, bins=50)
        plt.xlabel("Weight")
        plt.ylabel("Count")
        plt.title("Histogram of Sample Weights")
        plt.yscale("log")
        # plt.show()

        sorted_weights = np.sort(self.samples_weights_GP)[::-1]
        cum_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)

        plt.plot(cum_weights)
        plt.xlabel("Number of top samples")
        plt.ylabel("Cumulative weight fraction")
        plt.title("Cumulative Weight Distribution")
        # plt.show()


def main():
    method_plots = MethodPlots(
        id="0001",
        N_MAX=7,
        T=100,
        T0_REF=17,
        num_samples=1000,
        include_Mf=True,
        include_chif=True,
        use_nonlinear_params=False,
        decay_corrected=True,
    )

    # method_plots.qnm_list += [(3,2,0,1)]
    # method_plots._initialize_results()

    method_plots.load_tuned_parameters()
    method_plots.compute_mf_chif()
    method_plots.compute_quantities()

    # method_plots.diagnostic_plots()

    method_plots.plot_spirals_ani(show=False)
    method_plots.plot_spirals_static(output_path="outputs/amplitude_spiral.pdf", show=False)
    # method_plots.plot_spirals_static_320(output_path="outputs/amplitude_spiral_320+.pdf", show=False)
    method_plots.plot_mismatch(output_path="outputs/mismatch.pdf", show=False)
    method_plots.plot_log_likelihood(output_path="outputs/log_likelihood_mismatch_plot.pdf", show=False)
    method_plots.plot_amplitude(output_path="outputs/amplitude.pdf", show=False)
    method_plots.plot_significance(output_path="outputs/significance_320+.pdf", show=False)


def main_iter():
    for id, T0_ref in zip(["0001", "0002", "0003", "0004"], [17, 21, 23, 26]):
        for extra_qnm in [[], [(3, 2, 0, 1)]]:
            method_plots = MethodPlots(id=id, N_MAX=7, T=100, T0_REF=T0_ref, include_Mf=True, include_chif=True)
            method_plots.qnm_list += extra_qnm
            method_plots._initialize_results()

            method_plots.load_tuned_parameters()
            method_plots.compute_mf_chif()
            method_plots.compute_quantities()

            method_plots.plot_mismatch(output_path=f"outputs/mismatch_{id}_{extra_qnm}.pdf", show=False)
            method_plots.plot_log_likelihood(f"outputs/log_likelihood_mismatch_plot_{id}_{extra_qnm}.pdf", show=False)
            method_plots.plot_amplitude(output_path=f"outputs/amplitude_{id}_{extra_qnm}.pdf", show=False)
            method_plots.plot_significance(output_path=f"outputs/significance_{id}_{extra_qnm}.pdf", show=False)


if __name__ == "__main__":
    main()
