import qnmfits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from CCE import SXS_CCE
import bgp_qnm_fits as bgp

from scipy.optimize import minimize
from scipy import stats

from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot_config import PlotConfig


class MethodPlots2:

    config = PlotConfig()
    config.apply_style()

    def __init__(
        self,
        id,
        N_MAX=7,
        T=100,
        T0=np.linspace(-3,0,5),
        num_samples=10000,
        include_Mf=True,
        include_chif=True,
    ):
        self.id = id
        self.N_MAX = N_MAX
        self.T = T
        self.T0s = T0
        self.include_Mf = include_Mf
        self.include_chif = include_chif
        self.num_samples = num_samples

        self.sim_main = SXS_CCE(id, lev="Lev5", radius="R2")
        self.sim_lower = SXS_CCE(id, lev="Lev4", radius="R2")

        self.qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)]
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

        ref_fit_GP = bgp.BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
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
        return constant_term + np.einsum("p,pst->st", mean_vector - ref_params, model_terms)

    def ppc(self, output_path="outputs/ppc.pdf", show=False):
        """
        Generate posterior predictive checks.
        """
        fig2, ax2 = plt.subplots(1, 1, figsize=(self.config.fig_width, self.config.fig_height))
        colors = self.custom_colormap2(np.linspace(0, 1, len(self.T0s)))

        fit_times = self.T0s

        dof = 2 * len(self.spherical_modes) * len(self.fit_GP.fits[0]["analysis_times"]) 

        ax2.hist(stats.chi2(dof).rvs(self.num_samples) / dof, bins=50, alpha=0.5, color="gray")

        for i, fit_time in enumerate(fit_times):

            fit = self.fit_GP.fits[i]

            samples = fit["samples"]
            analysis_times = fit["analysis_times"]

            sample_models = np.zeros(
                (min(100, len(samples)), len(analysis_times)), dtype=np.complex128
            )
            chi_squareds = np.zeros(min(100, len(samples)))
            for j in range(min(100, len(samples))):
                theta_j = samples[j, :]
                sample_model = self.get_model_linear(
                    fit["constant_term"], theta_j, fit["ref_params"], fit["model_terms"]
                )
                residual = fit["data_array_masked"] - sample_model
                chi_squared = np.einsum(
                    "st,su,stu->",
                    np.conj(residual),
                    residual,
                    fit["inv_noise_covariance"],
                )
                chi_squareds[j] = chi_squared
                sample_models[j, :] = np.abs(sample_model.real)

            median_chi2 = np.median(chi_squareds / dof)
            ci_lower = np.percentile(chi_squareds / dof, 2.5)  # 2.5th percentile for 95% CI
            ci_upper = np.percentile(chi_squareds / dof, 97.5)  # 97.5th percentile for 95% CI
            ax2.axvline(x=median_chi2, color=colors[i], label=f"T0 = {fit_time:.2f}")
            ax2.axvspan(ci_lower, ci_upper, alpha=0.1, color=colors[i])

        ax2.set_xlabel("$\chi^2$")
        ax2.set_ylabel("Frequency")
        # Create a colorbar for the range of T0 times
        sm = plt.cm.ScalarMappable(cmap=self.custom_colormap2, norm=plt.Normalize(min(self.T0s), max(self.T0s)))
        sm.set_array([])  # Empty array for the data range
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('$T_0$', rotation=0, labelpad=10, fontsize=8)
        cbar.ax.tick_params(labelsize=6)

        fig2.savefig(output_path.replace(".pdf", "_chi2.pdf"), bbox_inches="tight")
        if show:
            plt.figure(fig2.number)
            plt.show()
        plt.close(fig2)


def main():
    method_plots = MethodPlots2(
        id="0001",
        N_MAX=6,
        T=100,
        num_samples=int(1e3),
        include_Mf=True,
        include_chif=True,
    )

    method_plots.load_tuned_parameters()
    method_plots.get_t0_ref_fits()
    method_plots.ppc(output_path="outputs/ppc.pdf", show=False)


if __name__ == "__main__":
    main()