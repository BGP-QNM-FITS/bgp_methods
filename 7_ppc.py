import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import bgp_qnm_fits as bgp

from matplotlib.colors import to_hex
from matplotlib.colors import LinearSegmentedColormap

from plot_config import PlotConfig
from scipy.interpolate import interp1d


class MethodPlots2:

    config = PlotConfig()
    config.apply_style()

    def __init__(
        self,
        id,
        N_MAX=7,
        T=100,
        T0=np.linspace(-5, 5, 20),
        num_samples=10000,
        include_Mf=True,
        include_chif=True,
        data_type='news',
        strain_parameters=True, 
    ):
        self.id = id
        self.N_MAX = N_MAX
        self.T = T
        self.T0s = T0
        self.include_Mf = include_Mf
        self.include_chif = include_chif
        self.num_samples = num_samples
        self.data_type = data_type
        self.strain_parameters = strain_parameters

        self.sim_main = bgp.SXS_CCE(id, type=self.data_type, lev="Lev5", radius="R2")
        self.sim_lower = bgp.SXS_CCE(id, type=self.data_type, lev="Lev4", radius="R2")

        self.qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)] + [(3, 2, 0, 1)]
        self.spherical_modes = [(2, 2)]

        self.chif_mag_ref = self.sim_main.chif_mag
        self.Mf_ref = self.sim_main.Mf

        colors = self.config.colors
        colors2 = self.config.colors2
        self.custom_colormap = LinearSegmentedColormap.from_list("custom_colormap", colors)
        self.custom_colormap2 = LinearSegmentedColormap.from_list("custom_colormap2", colors2)
        self.fundamental_color_WN = to_hex("#395470")
        self.fundamental_color_GP = to_hex("#395471")
        self.overtone_color_WN = to_hex("#65858c")
        self.overtone_color_GP = to_hex("#65858d")

    def load_tuned_parameters(self):
        """
        Load tuned kernel parameters for GP and WN fits.
        """
        self.tuned_param_dict_GP = bgp.get_param_data("GP")[self.id]
        self.tuned_param_dict_WN = bgp.get_param_data("WN")[self.id]

    def get_t0_ref_fits(self):
        """
        Get the fits for the reference t0 using class variables.
        """

        # dt = 0.1
        # sim_times_interp = np.arange(self.sim_main.times[0], self.sim_main.times[-1] + dt, dt)
        # sim_h_interp = bgp.sim_interpolator_data(self.sim_main.h, self.sim_main.times, sim_times_interp)

        ref_fit_GP = bgp.BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            # sim_times_interp,
            # sim_h_interp,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.tuned_param_dict_GP,
            bgp.kernel_GP,
            t0=self.T0s,
            use_nonlinear_params=False,
            num_samples=self.num_samples,
            t0_method="geq",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
            strain_parameters=self.strain_parameters,
            data_type=self.data_type,
        )

        self.fit_GP = ref_fit_GP

    def get_model_linear(self, constant_term, mean_vector, ref_params, model_terms):
        """
        Compute the linear model array based on the mean vector and model terms.
        Args:
            constant_term (float): The constant term in the model.
            mean_vector (array): The mean vector of the parameters.
            ref_params (array): Reference parameters for the model.
            model_terms (array): The model terms.
        Returns:
            model_array (array): The linear model array.
        """
        return constant_term + np.einsum("p,stp->st", mean_vector - ref_params, model_terms)

    def ppc_main(self, output_path="outputs/ppc.pdf", show=False):
        """
        Generate posterior predictive checks.
        """

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(self.config.fig_width, self.config.fig_height * 1.5), gridspec_kw={"height_ratios": [1, 2]}
        )
        dummy_fig, dummy_ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))
        colors = self.custom_colormap2(np.linspace(0, 1, len(self.T0s)))
        fit_times = self.T0s

        cdf_lefts_median = np.zeros(len(fit_times))
        cdf_lefts_upper = np.zeros(len(fit_times))
        cdf_lefts_lower = np.zeros(len(fit_times))

        for i, fit_time in enumerate(fit_times):

            fit = self.fit_GP.fits[i]

            samples = fit["samples"]
            r_squareds = np.zeros(min(1000, len(samples)))

            eigvals = np.linalg.eigvals(fit["noise_covariance"])[0].real
            eigvals = eigvals[eigvals > 1e-11]
            num_draws = int(1e4)
            normal_samples = np.random.normal(0, 1, size=(num_draws, len(eigvals)))
            dist_samples = np.sum(eigvals * normal_samples**2, axis=1)

            kde = sns.kdeplot(dist_samples, ax=dummy_ax, color=colors[i], alpha=0.5, bw_adjust=3)
            cdf = np.cumsum(kde.get_lines()[-1].get_ydata())
            cdf = cdf / cdf[-1]
            x_values = kde.get_lines()[-1].get_xdata()

            cdf_vals = []

            for j in range(min(1000, len(samples))):
                theta_j = samples[j, :]
                sample_model = self.get_model_linear(
                    fit["constant_term"], theta_j, fit["ref_params"], fit["model_terms"]
                )
                residual = fit["data_array_masked"] - sample_model
                r_squared = np.einsum("st, st -> ", np.conj(residual), residual).real
                r_squareds[j] = r_squared

                index = np.argmin(np.abs(x_values - r_squared))
                cdf_val = cdf[index]
                cdf_vals.append(cdf_val)

            median_chi2 = np.median(r_squareds)
            ci_lower = np.percentile(r_squareds, 25)
            ci_upper = np.percentile(r_squareds, 75)

            cdf_lefts_median[i] = np.median(cdf_vals)
            cdf_lefts_upper[i] = np.percentile(cdf_vals, 75)
            cdf_lefts_lower[i] = np.percentile(cdf_vals, 25)

            ax1.set_ylabel(r"$\mathrm{CDF}$")
            ax1.set_xlabel(r"$t_0 [M]$")

            t0_choices = [1, 2, 3, 5]
            closest_indices = [np.argmin(np.abs(fit_times - t0_choice)) for t0_choice in t0_choices]
            if fit_time in fit_times[closest_indices]:
                sns.kdeplot(dist_samples, ax=ax2, color=colors[i], alpha=0.6, bw_adjust=1)
                ax2.axvline(x=median_chi2, color=colors[i])
                ax2.axvspan(ci_lower, ci_upper, alpha=0.1, color=colors[i])
                ax2.text(
                    median_chi2 - 0.0001,
                    2593.4840329013887,
                    rf"$t_0={fit_time:.2f} \, [M]$",
                    color="k",
                    rotation=90,
                    ha="center",
                    va="top",
                )
                ax1.plot(fit_time, cdf_lefts_median[i], marker="o", color=colors[i], markersize=3, zorder=10)

        fit_times_dense = np.linspace(fit_times[0], fit_times[-1], 500)
        cdf_lefts_median_interp = interp1d(fit_times, cdf_lefts_median, kind="cubic")
        cdf_lefts_median_smooth = cdf_lefts_median_interp(fit_times_dense)
        cdf_lefts_upper_interp = interp1d(fit_times, cdf_lefts_upper, kind="cubic")
        cdf_lefts_upper_smooth = cdf_lefts_upper_interp(fit_times_dense)
        cdf_lefts_lower_interp = interp1d(fit_times, cdf_lefts_lower, kind="cubic")
        cdf_lefts_lower_smooth = cdf_lefts_lower_interp(fit_times_dense)

        ax1.axhline(0.5, color="k", linestyle="-", alpha=0.4)
        ax1.plot(fit_times_dense, cdf_lefts_median_smooth, color="k", linestyle="-")
        ax1.fill_between(fit_times_dense, cdf_lefts_lower_smooth, cdf_lefts_upper_smooth, color="k", alpha=0.1)
        ax1.set_xlim(0, 7)
        ax1.set_ylim(0, 1.05)
        ax2.set_xlabel(r"$\xi^2$")
        ax2.set_ylabel("Relative frequency")
        # ax2.set_xlim(0.89, 1.11)
        ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax2.set_xticks(np.linspace(0, 0.003, 4))

        fig.savefig(output_path, bbox_inches="tight")
        if show:
            plt.figure(fig.number)
            plt.show()
        plt.close(fig)


def main():
    method_plots = MethodPlots2(
        id="0001",
        N_MAX=6,
        T=100,
        # T0=np.linspace(-4,-3,6),
        T0=np.arange(0, 7.1, 0.5),
        # T0=np.linspace(-4, -3, 5),
        num_samples=int(1e3),
        include_Mf=True,
        include_chif=True,
        data_type='news',
        strain_parameters=False, 
    )

    method_plots.load_tuned_parameters()
    method_plots.get_t0_ref_fits()
    method_plots.ppc_main(output_path="outputs/ppc.pdf", show=False)


if __name__ == "__main__":
    main()
