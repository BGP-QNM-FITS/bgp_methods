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


class MethodPlots2:

    config = PlotConfig()
    config.apply_style()

    def __init__(
        self,
        id,
        N_MAX=7,
        T=100,
        T0_REF=17,
        num_samples=10000,
        large_num_samples=100000,
        include_Mf=True,
        include_chif=True,
    ):
        self.id = id
        self.N_MAX = N_MAX
        self.T = T
        self.T0_REF = T0_REF
        self.include_Mf = include_Mf
        self.include_chif = include_chif
        self.num_samples = num_samples
        self.large_num_samples = large_num_samples

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

    def get_t0_ref_fits(self):
        """
        Get the fits for the reference t0 using class variables.
        """

        ref_fit_LS = qnmfits.multimode_ringdown_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_t0,
            self.chif_t0,
            self.T0_REF,
            T=self.T,
            spherical_modes=self.spherical_modes,
        )

        ref_fit_WN = bgp.BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.tuned_param_dict_WN,
            bgp.kernel_WN,
            t0=self.T0_REF,
            use_nonlinear_params=False,
            num_samples=self.num_samples,
            t0_method="geq",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
        )

        ref_fit_GP = bgp.BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.tuned_param_dict_GP,
            bgp.kernel_GP,
            t0=self.T0_REF,
            use_nonlinear_params=False,
            num_samples=self.num_samples,
            t0_method="geq",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
        )

        ref_fit_GP_large_sample = bgp.BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.tuned_param_dict_GP,
            bgp.kernel_GP,
            t0=self.T0_REF,
            use_nonlinear_params=False,
            num_samples=self.large_num_samples,
            t0_method="geq",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
        )

        ref_fit_WN_large_sample = bgp.BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.tuned_param_dict_WN,
            bgp.kernel_WN,
            t0=self.T0_REF,
            use_nonlinear_params=False,
            num_samples=self.large_num_samples,
            t0_method="geq",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
        )

        ref_fit_WN_NL = bgp.BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.tuned_param_dict_WN,
            bgp.kernel_WN,
            t0=self.T0_REF,
            use_nonlinear_params=True,
            num_samples=self.num_samples,
            t0_method="geq",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
        )

        ref_fit_GP_NL = bgp.BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.tuned_param_dict_GP,
            bgp.kernel_GP,
            t0=self.T0_REF,
            use_nonlinear_params=True,
            num_samples=self.num_samples,
            t0_method="geq",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
        )

        self.fit_WN = ref_fit_WN
        self.fit_GP = ref_fit_GP
        self.fit_WN_NL = ref_fit_WN_NL
        self.fit_GP_NL = ref_fit_GP_NL

        self.ref_params = []
        for re_c, im_c in zip(np.real(ref_fit_LS["C"]), np.imag(ref_fit_LS["C"])):
            self.ref_params.append(re_c)
            self.ref_params.append(im_c)

        self.ref_samples_WN = ref_fit_WN.fit["samples"]
        self.ref_samples_GP = ref_fit_GP.fit["samples"]

        self.ref_samples_WN_NL = ref_fit_WN_NL.fit["samples"]
        self.ref_samples_GP_NL = ref_fit_GP_NL.fit["samples"]

        self.samples_abs_WN = ref_fit_WN_large_sample.fit["sample_amplitudes"]
        self.samples_abs_GP = ref_fit_GP_large_sample.fit["sample_amplitudes"]
        self.samples_weights_WN = ref_fit_WN_large_sample.fit["samples_weights"]
        self.samples_weights_GP = ref_fit_GP_large_sample.fit["samples_weights"]

        self.residual_WN = np.zeros((len(ref_fit_WN.fit["analysis_times"])))
        self.residual_GP = np.zeros((len(ref_fit_GP.fit["analysis_times"])))

        mode_index = self.spherical_modes.index((2, 2))

        self.model_times_WN = ref_fit_WN.fit["analysis_times"]
        self.model_times_GP = ref_fit_GP.fit["analysis_times"]

        self.param_list = [qnm for qnm in self.qnm_list for _ in range(2)] + ["chif"] + ["Mf"]

        # Get various quantities

        self.significance_GP = bgp.get_significance_list(
            self.qnm_list, ref_fit_GP.fit["mean"], ref_fit_GP.fit["fisher_matrix"], include_chif=True, include_Mf=True
        )
        self.significance_WN = bgp.get_significance_list(
            self.qnm_list, ref_fit_WN.fit["mean"], ref_fit_WN.fit["fisher_matrix"], include_chif=True, include_Mf=True
        )

        self.neff_GP = ref_fit_GP_large_sample.fit["N_effective_samples"]
        self.neff_WN = ref_fit_WN_large_sample.fit["N_effective_samples"]

        print("N_eff GP:", self.neff_GP)
        print("N_eff WN:", self.neff_WN)

        print("Significance GP:", self.significance_GP)
        print("Significance WN:", self.significance_WN)

    def plot_fundamental_kde(self, output_path="outputs/fundamental_corner.pdf", show=False):

        parameter_choice = [(2, 2, 0, 1)]

        labels = [
            (
                rf"$\mathrm{{Re}}(C_{{({param[0]},{param[1]},{param[2]},+)}})$"
                if i % 2 == 0
                else rf"$\mathrm{{Im}}(C_{{({param[0]},{param[1]},{param[2]},+)}})$"
            )
            for param in parameter_choice
            for i in range(2)
        ]

        abs_indices_fundamental = [i for i, param in enumerate(self.qnm_list) if param in parameter_choice]
        indices_fundamental = [i for i, param in enumerate(self.param_list) if param in parameter_choice]

        samples_fundamental_WN = self.ref_samples_WN[:, indices_fundamental]
        samples_fundamental_GP = self.ref_samples_GP[:, indices_fundamental]

        samples_abs_fundamental_WN = self.samples_abs_WN[:, abs_indices_fundamental]
        samples_abs_fundamental_GP = self.samples_abs_GP[:, abs_indices_fundamental]

        df_samples_fundamental_WN = pd.DataFrame(samples_fundamental_WN, columns=labels)
        df_samples_fundamental_GP = pd.DataFrame(samples_fundamental_GP, columns=labels)

        df_samples_fundamental_WN["Dataset"] = "WN"
        df_samples_fundamental_GP["Dataset"] = "GP"

        # Create the jointplot
        g = sns.jointplot(
            data=df_samples_fundamental_GP,
            x=labels[0],
            y=labels[1],
            hue="Dataset",
            kind="kde",
            palette=[self.fundamental_color_GP],
            marginal_kws={"fill": False},
            height=self.config.fig_width,
            levels=[0.1, 0.5],
        )

        # Adjust the linewidth of the KDE plots
        for collection in g.ax_joint.collections:
            collection.set_linewidth(1.5)  # Set the desired linewidth

        # Adjust the linewidth of the KDE plots in the marginal plots
        for line in g.ax_marg_x.lines:
            line.set_linewidth(1.5)  # Set the desired linewidth
        for line in g.ax_marg_y.lines:
            line.set_linewidth(1.5)  # Set the desired linewidth

        sns.kdeplot(
            df_samples_fundamental_WN[labels[0]],
            ax=g.ax_marg_x,
            color=self.fundamental_color_WN,
            linestyle="--",
            fill=False,
            linewidth=1.5,
        )
        sns.kdeplot(
            y=df_samples_fundamental_WN[labels[1]],
            ax=g.ax_marg_y,
            color=self.fundamental_color_WN,
            linestyle="--",
            fill=False,
            linewidth=1.5,
        )
        sns.kdeplot(
            x=df_samples_fundamental_WN[labels[0]],
            y=df_samples_fundamental_WN[labels[1]],
            ax=g.ax_joint,
            color=self.fundamental_color_WN,
            fill=False,
            levels=[0.1, 0.5],
            linewidths=1.5,
        )

        g.ax_joint.legend_.remove()

        g.ax_joint.plot(
            self.ref_params[indices_fundamental[0]],
            self.ref_params[indices_fundamental[1]],
            "*",
            color="#DE6A5E",
            markersize=10,
        )

        # Get dashed lines for the WN contours
        for collection in g.ax_joint.collections:
            color = None
            if collection.get_edgecolor().size:
                color = to_hex(collection.get_edgecolor()[0])
            elif collection.get_facecolor().size:
                color = to_hex(collection.get_facecolor()[0])
            if color == self.fundamental_color_WN:
                collection.set_linestyle("--")  # Set linestyle to dashed for WN
                collection.set_linewidth(1.5)  # Set linewidth for WN contours

        # Add inset plot
        ax_inset = inset_axes(
            g.ax_joint,
            width="50%",
            height="20%",
            loc="lower right",
            borderpad=1,
            bbox_to_anchor=(0, 0.03, 1, 1),
            bbox_transform=g.ax_joint.transAxes,
        )

        df_samples_abs_fundamental_WN = pd.DataFrame(
            {"Amplitude": samples_abs_fundamental_WN.flatten(), "Dataset": "WN"}
        )
        df_samples_abs_fundamental_GP = pd.DataFrame(
            {"Amplitude": samples_abs_fundamental_GP.flatten(), "Dataset": "GP"}
        )

        df_samples_abs_fundamental_WN["Weight"] = self.samples_weights_WN
        df_samples_abs_fundamental_GP["Weight"] = self.samples_weights_GP

        sns.kdeplot(
            data=df_samples_abs_fundamental_GP,
            x="Amplitude",
            color=self.fundamental_color_GP,
            label="GP (Prior 1)",
            linewidth=1.5,
            ax=ax_inset,
        )
        sns.kdeplot(
            data=df_samples_abs_fundamental_WN,
            x="Amplitude",
            color=self.fundamental_color_WN,
            linestyle="--",
            linewidth=1.5,
            label="WN (Prior 1)",
            ax=ax_inset,
        )

        sns.kdeplot(
            data=df_samples_abs_fundamental_GP,
            x="Amplitude",
            color=self.fundamental_color_GP,
            label="GP (Prior 2)",
            linewidth=0.5,
            weights="Weight",
            ax=ax_inset,
        )
        sns.kdeplot(
            data=df_samples_abs_fundamental_WN,
            x="Amplitude",
            color=self.fundamental_color_WN,
            label="WN (Prior 2)",
            linestyle="--",
            linewidth=0.5,
            weights="Weight",
            ax=ax_inset,
        )

        ax_inset.set_title(r"$|C_{\alpha}|$", fontsize=8)
        # ax_inset.set_xlim(0.19, 0.23)
        # ax_inset.set_ylim(0.0, 300)
        ax_inset.set_ylabel("")
        ax_inset.set_xlabel("")
        ax_inset.set_yticklabels([])
        ax_inset.yaxis.set_ticks([])
        ax_inset.tick_params(axis="both", which="major", labelsize=6)

        line_styles_inset = [
            Line2D(
                [0],
                [0],
                color=self.fundamental_color_GP,
                linewidth=1.5,
                label="Prior 1",
            ),
            Line2D(
                [0],
                [0],
                color=self.fundamental_color_WN,
                linewidth=0.5,
                label="Prior 2",
            ),
        ]

        ax_inset.legend(
            handles=line_styles_inset,
            loc="upper left",
            frameon=False,
            ncol=1,
            fontsize=4,
        )

        # g.ax_joint.set_xlim(-0.175, -0.135)
        g.ax_joint.set_ylim(0.01, 0.065)

        line_styles = [
            Line2D(
                [0],
                [0],
                color=self.fundamental_color_GP,
                linestyle="-",
                label="GP",
                linewidth=1.5,
            ),
            Line2D(
                [0],
                [0],
                color=self.fundamental_color_WN,
                linestyle="--",
                label="WN",
                linewidth=1.5,
            ),
        ]

        g.figure.legend(
            handles=line_styles,
            loc="upper left",
            frameon=False,
            bbox_to_anchor=(0.22, 0.84),
            ncol=1,
            fontsize=7,
        )

        g.figure.savefig(output_path, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(g.figure)

    def plot_overtone_kde(self, output_path="outputs/overtone_corner.pdf", show=False):

        parameter_choice = [(2, 2, 2, 1)]

        labels = [
            (
                rf"$\mathrm{{Re}}(C_{{({param[0]},{param[1]},{param[2]},+)}})$"
                if i % 2 == 0
                else rf"$\mathrm{{Im}}(C_{{({param[0]},{param[1]},{param[2]},+)}})$"
            )
            for param in parameter_choice
            for i in range(2)
        ]

        abs_indices_overtone = [i for i, param in enumerate(self.qnm_list) if param in parameter_choice]
        indices_overtone = [i for i, param in enumerate(self.param_list) if param in parameter_choice]

        samples_overtone_WN = self.ref_samples_WN[:, indices_overtone]
        samples_overtone_GP = self.ref_samples_GP[:, indices_overtone]

        samples_abs_overtone_WN = self.samples_abs_WN[:, abs_indices_overtone]
        samples_abs_overtone_GP = self.samples_abs_GP[:, abs_indices_overtone]

        df_samples_overtone_WN = pd.DataFrame(samples_overtone_WN, columns=labels)
        df_samples_overtone_GP = pd.DataFrame(samples_overtone_GP, columns=labels)

        df_samples_overtone_WN["Dataset"] = "WN"
        df_samples_overtone_GP["Dataset"] = "GP"

        # Create the jointplot
        g = sns.jointplot(
            data=df_samples_overtone_GP,
            x=labels[0],
            y=labels[1],
            hue="Dataset",
            kind="kde",
            palette=[self.overtone_color_GP],
            marginal_kws={"fill": False},
            height=self.config.fig_width,
            levels=[0.1, 0.5],
        )

        # Adjust the linewidth of the KDE plots in the joint plot
        for collection in g.ax_joint.collections:
            collection.set_linewidth(1.5)  # Set the desired linewidth

        # Adjust the linewidth of the KDE plots in the marginal plots
        for line in g.ax_marg_x.lines:
            line.set_linewidth(1.5)  # Set the desired linewidth
        for line in g.ax_marg_y.lines:
            line.set_linewidth(1.5)  # Set the desired linewidth

        sns.kdeplot(
            df_samples_overtone_WN[labels[0]],
            ax=g.ax_marg_x,
            color=self.overtone_color_WN,
            linestyle="--",
            fill=False,
            linewidth=1.5,
        )
        sns.kdeplot(
            y=df_samples_overtone_WN[labels[1]],
            ax=g.ax_marg_y,
            color=self.overtone_color_WN,
            linestyle="--",
            fill=False,
            linewidth=1.5,
        )
        sns.kdeplot(
            x=df_samples_overtone_WN[labels[0]],
            y=df_samples_overtone_WN[labels[1]],
            ax=g.ax_joint,
            color=self.overtone_color_WN,
            fill=False,
            levels=[0.1, 0.5],
            linewidth=1.5,
        )

        g.ax_joint.legend_.remove()

        g.ax_joint.plot(
            self.ref_params[indices_overtone[0]],
            self.ref_params[indices_overtone[1]],
            "*",
            color="#DE6A5E",
            markersize=10,
        )

        # Get dashed lines for the WN contours
        for collection in g.ax_joint.collections:
            color = None
            if collection.get_edgecolor().size:
                color = to_hex(collection.get_edgecolor()[0])
            elif collection.get_facecolor().size:
                color = to_hex(collection.get_facecolor()[0])
            if color == self.overtone_color_WN:
                collection.set_linestyle("--")  # Set linestyle to dashed for WN
                collection.set_linewidth(1.5)  # Set linewidth for WN contours

        g.ax_joint.axvline(0, color="black", linestyle=":", linewidth=1)
        g.ax_joint.axhline(0, color="black", linestyle=":", linewidth=1)

        ax_inset = inset_axes(
            g.ax_joint,
            width="50%",
            height="20%",
            loc="lower right",
            borderpad=1,
            bbox_to_anchor=(0, 0.03, 1, 1),
            bbox_transform=g.ax_joint.transAxes,
        )

        df_samples_abs_overtone_WN = pd.DataFrame(
            {
                "Amplitude": np.hstack(
                    (
                        samples_abs_overtone_WN.flatten(),
                        -samples_abs_overtone_WN.flatten(),
                    )
                ),
                "Dataset": "WN",
            }
        )
        df_samples_abs_overtone_GP = pd.DataFrame(
            {
                "Amplitude": np.hstack(
                    (
                        samples_abs_overtone_GP.flatten(),
                        -samples_abs_overtone_GP.flatten(),
                    )
                ),
                "Dataset": "GP",
            }
        )

        df_samples_abs_overtone_WN["Weight"] = np.hstack((self.samples_weights_WN, self.samples_weights_WN))
        df_samples_abs_overtone_GP["Weight"] = np.hstack((self.samples_weights_GP, self.samples_weights_GP))

        sns.kdeplot(
            data=df_samples_abs_overtone_GP,
            x="Amplitude",
            color=self.overtone_color_GP,
            linewidth=1.5,
            label="GP (Prior 1)",
            ax=ax_inset,
        )
        sns.kdeplot(
            data=df_samples_abs_overtone_WN,
            x="Amplitude",
            color=self.overtone_color_WN,
            linestyle="--",
            linewidth=1.5,
            label="WN (Prior 1)",
            ax=ax_inset,
        )

        sns.kdeplot(
            data=df_samples_abs_overtone_GP,
            x="Amplitude",
            color=self.overtone_color_GP,
            label="GP (Prior 2)",
            linewidth=0.5,
            weights="Weight",
            ax=ax_inset,
        )
        sns.kdeplot(
            data=df_samples_abs_overtone_WN,
            x="Amplitude",
            color=self.overtone_color_WN,
            label="WN (Prior 2)",
            linestyle="--",
            linewidth=0.5,
            weights="Weight",
            ax=ax_inset,
        )

        ax_inset.set_title(r"$|C_{\alpha}|$", fontsize=8)
        ax_inset.set_xlim(0, 2)
        # ax_inset.set_ylim(0, 2.5)
        ax_inset.set_ylabel("")
        ax_inset.set_xlabel("")
        ax_inset.set_yticklabels([])
        ax_inset.yaxis.set_ticks([])
        ax_inset.tick_params(axis="both", which="major", labelsize=6)

        line_styles_inset = [
            Line2D([0], [0], color=self.overtone_color_GP, linewidth=1.5, label="Prior 1"),
            Line2D([0], [0], color=self.overtone_color_WN, linewidth=0.5, label="Prior 2"),
        ]

        ax_inset.legend(
            handles=line_styles_inset,
            loc="upper right",
            frameon=False,
            ncol=1,
            fontsize=6,
        )

        g.ax_joint.set_ylim(-4, 2)

        line_styles = [
            Line2D(
                [0],
                [0],
                color=self.overtone_color_GP,
                linestyle="-",
                label="GP",
                linewidth=1.5,
            ),
            Line2D(
                [0],
                [0],
                color=self.overtone_color_WN,
                linestyle="--",
                label="WN",
                linewidth=1.5,
            ),
        ]

        g.figure.legend(
            handles=line_styles,
            loc="upper left",
            frameon=False,
            bbox_to_anchor=(0.22, 0.84),
            ncol=1,
            fontsize=7,
        )

        g.figure.savefig(output_path, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(g.figure)

    def plot_mass_spin_corner(self, output_path="outputs/mass_spin_corner.pdf", show=False):
        parameter_choice = ["chif", "Mf"]

        indices_Chif_M = [i for i, param in enumerate(self.param_list) if param in parameter_choice]
        labels_Chif_M = parameter_choice

        samples_Chif_M_WN = self.ref_samples_WN[:, indices_Chif_M]
        samples_Chif_M_GP = self.ref_samples_GP[:, indices_Chif_M]

        df_wn_Chif_M = pd.DataFrame(samples_Chif_M_WN, columns=labels_Chif_M)
        df_main_Chif_M = pd.DataFrame(samples_Chif_M_GP, columns=labels_Chif_M)

        df_wn_Chif_M["Dataset"] = "WN"
        df_main_Chif_M["Dataset"] = "GP"

        # Create the jointplot
        g = sns.jointplot(
            data=df_main_Chif_M,
            x="chif",
            y="Mf",
            hue="Dataset",
            kind="kde",
            palette=[self.fundamental_color_GP],
            marginal_kws={"fill": False},
            height=self.config.fig_width,
            levels=[0.1, 0.5],
        )

        sns.kdeplot(
            df_wn_Chif_M[labels_Chif_M[0]],
            ax=g.ax_marg_x,
            color=self.fundamental_color_WN,
            linestyle="--",
            fill=False,
        )
        sns.kdeplot(
            y=df_wn_Chif_M[labels_Chif_M[1]],
            ax=g.ax_marg_y,
            color=self.fundamental_color_WN,
            linestyle="--",
            fill=False,
        )
        sns.kdeplot(
            x=df_wn_Chif_M[labels_Chif_M[0]],
            y=df_wn_Chif_M[labels_Chif_M[1]],
            ax=g.ax_joint,
            color=self.fundamental_color_WN,
            fill=False,
            levels=[0.1, 0.5],
        )

        # --- Adjust the central plot (ax_joint) KDEs ---
        for collection in g.ax_joint.collections:
            color = None
            if collection.get_edgecolor().size:
                color = to_hex(collection.get_edgecolor()[0])
            elif collection.get_facecolor().size:
                color = to_hex(collection.get_facecolor()[0])

            if color == self.fundamental_color_WN:
                collection.set_linestyle("--")  # Set linestyle to dashed for WN

        # Add vertical and horizontal dotted lines at the truth values
        g.ax_joint.plot(self.chif_t0, self.Mf_t0, "*", color="#DE6A5E", markersize=10)
        g.ax_joint.plot(self.chif_mag_ref, self.Mf_ref, "x", color="#DE6A5E", markersize=10)

        # Add legend for the truth values
        g.ax_joint.legend(loc="upper right", frameon=False)

        # Add labels
        g.set_axis_labels(r"$\chi_f$", r"$M_f$")

        g.ax_joint.set_xlim(0.665, 0.71)
        g.ax_joint.set_ylim(0.937, 0.97)

        line_styles = [
            Line2D([0], [0], color=self.fundamental_color_GP, linestyle="-", label="GP"),
            Line2D([0], [0], color=self.fundamental_color_WN, linestyle="--", label="WN"),
        ]

        g.figure.legend(
            handles=line_styles,
            loc="upper left",
            frameon=False,
            bbox_to_anchor=(0.2, 0.83),
        )

        g.figure.savefig(output_path, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(g.figure)

    def linear_approximation(self, output_path="outputs/linear_approximation.pdf", show=False):

        fig, ax = plt.subplots(1, 1, figsize=(self.config.fig_width, self.config.fig_height))
        colors = self.custom_colormap2(np.linspace(0, 1, 3))

        fits = [self.fit_GP, self.fit_WN, self.fit_GP_NL, self.fit_WN_NL]
        fitnames = ["GP", "WN", "GP_NL", "WN_NL"]

        ell, m, n, sign = (2, 2, 0, 1)  # Note this is currently hardcoded below!!
        ellp, mp = (2, 2)

        # Get indices for the chosen modes
        qnm_index = self.qnm_list.index((ell, m, n, sign))
        spherical_mode_index = self.spherical_modes.index((ellp, mp))

        for fit, fitname in zip(fits, fitnames):

            mass_ref = fit.fit["Mf_ref"]
            chif_ref = fit.fit["chif_ref"]

            freq_ref = fit.fit["frequencies"][qnm_index]
            dfreq_dchi_ref = fit.fit["frequency_derivatives"][qnm_index]

            mixing_ref = fit.fit["mixing_coefficients"][spherical_mode_index][qnm_index]
            dmu_dchi_ref = fit.fit["mixing_derivatives"][spherical_mode_index][qnm_index]

            chifs = fit.fit["samples"][:, -2]
            mfs = fit.fit["samples"][:, -1]

            real_freq_deviations = []
            imag_freq_deviations = []
            mixing_deviations = []

            for i, _ in enumerate(fit.fit["samples"]):

                nonlinear_freq = qnmfits.qnm.omega(ell, m, n, sign, chifs[i], Mf=mfs[i], s=-2)
                nonlinear_mixing = qnmfits.qnm.mu(ellp, mp, ell, m, n, sign, chifs[i])
                linear_freq = (
                    freq_ref - (mfs[i] - mass_ref) * freq_ref / mass_ref + (chifs[i] - chif_ref) * dfreq_dchi_ref
                )
                linear_mixing = mixing_ref + (chifs[i] - chif_ref) * dmu_dchi_ref

                real_freq_deviations.append(np.abs(np.real(nonlinear_freq - linear_freq)) / np.real(nonlinear_freq))
                imag_freq_deviations.append(np.abs(np.imag(nonlinear_freq - linear_freq)) / -np.imag(nonlinear_freq))
                mixing_deviations.append(np.abs(nonlinear_mixing - linear_mixing) / np.abs(nonlinear_mixing))

            real_freq_deviations = np.hstack((np.array(real_freq_deviations), -np.array(real_freq_deviations)))
            imag_freq_deviations = np.hstack((np.array(imag_freq_deviations), -np.array(imag_freq_deviations)))
            mixing_deviations = np.hstack((np.array(mixing_deviations), -np.array(mixing_deviations)))

            if fitname == "GP":
                ls = "-"
                lw = 1.0
            elif fitname == "WN":
                ls = "--"
                lw = 1.0
            elif fitname == "GP_NL":
                ls = "-"
                lw = 0.5
            elif fitname == "WN_NL":
                ls = "--"
                lw = 0.5

            sns.kdeplot(
                real_freq_deviations,
                ax=ax,
                color=colors[0],
                linestyle=ls,
                linewidth=lw,
                bw_adjust=3.0,
            )

            sns.kdeplot(
                imag_freq_deviations,
                ax=ax,
                color=colors[1],
                linestyle=ls,
                linewidth=lw,
                bw_adjust=3.0,
            )

            sns.kdeplot(
                mixing_deviations,
                ax=ax,
                color=colors[2],
                linestyle=ls,
                linewidth=lw,
                bw_adjust=3.0,
            )

        real_freq_line = Line2D([0], [0], color=colors[0], linestyle="-", linewidth=1.0)
        imag_freq_line = Line2D([0], [0], color=colors[1], linestyle="-", linewidth=1.0)
        mixing_line = Line2D([0], [0], color=colors[2], linestyle="-", linewidth=1.0)
        leg2 = ax.legend(
            [real_freq_line, imag_freq_line, mixing_line],
            [r"Re($\omega_{\alpha}$)", r"Im($\omega_{\alpha}$)", r"$|\mu_{\alpha}^{\beta}|$"],
            frameon=False,
            loc="lower left",
            ncol=1,
        )

        # Create a legend for line styles
        solid_line = Line2D([0], [0], color="black", linestyle="-", linewidth=1.0)
        dashed_line = Line2D([0], [0], color="black", linestyle="--", linewidth=1.0)
        thin_solid_line = Line2D([0], [0], color="black", linestyle="-", linewidth=0.5)
        thin_dashed_line = Line2D([0], [0], color="black", linestyle="--", linewidth=0.5)

        leg1 = ax.legend(
            [solid_line, dashed_line, thin_solid_line, thin_dashed_line],
            ["GP (ABD)", "WN (ABD)", "GP (nonlinear)", "WN (nonlinear)"],
            frameon=False,
            loc="upper right",
            ncol=1,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e2, 1e7)
        ax.set_xlim(1e-8, 1e-3)

        # Add both legends to the plot
        ax.add_artist(leg1)
        ax.add_artist(leg2)

        ax.set_xlabel("Fractional error")

        plt.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def ppc(self, output_path="outputs/ppc.pdf", show=False):
        """
        Generate posterior predictive checks.
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.config.fig_width, self.config.fig_height))
        fig2, ax2 = plt.subplots(1, 1, figsize=(self.config.fig_width, self.config.fig_height))
        colors = self.custom_colormap2(np.linspace(0, 1, 2))

        ax.plot(
            self.fit_GP.fit["analysis_times"],
            np.abs(self.fit_GP.fit["data_array_masked"][0].real),
            color="black",
            label="Data",
        )

        fits = [self.fit_GP, self.fit_WN]
        fitnames = ["GP", "WN"]

        for i, fit in enumerate(fits):
            sample_models = np.zeros(
                (min(100, len(fit.fit["samples"])), len(fit.fit["analysis_times"])), dtype=np.complex128
            )
            chi_squareds = np.zeros(min(100, len(fit.fit["samples"])))
            for j in range(min(100, len(fit.fit["samples"]))):
                theta_j = fit.fit["samples"][j, :]
                sample_model = fit.get_model_linear(
                    fit.fit["constant_term"], theta_j, fit.fit["ref_params"], fit.fit["model_terms"]
                )
                residual = fit.fit["data_array_masked"] - sample_model
                chi_squared = np.einsum(
                    "st,su,stu->",
                    np.conj(residual),
                    residual,
                    fit.fit["inv_noise_covariance"],
                )
                chi_squareds[j] = chi_squared
                sample_models[j, :] = np.abs(sample_model.real)

            # Calculate min and max instead of percentiles
            lower = np.min(sample_models, axis=0)
            upper = np.max(sample_models, axis=0)
            ax.fill_between(fit.fit["analysis_times"], lower, upper, color=colors[i], alpha=0.5, label=fitnames[i])

            label = "GP" if i == 0 else "WN"
            ax2.hist(chi_squareds, bins=50, alpha=0.7, color=colors[i], label=label)

        ax2.set_xlabel("$\chi^2$")
        ax2.set_ylabel("Frequency")
        ax2.legend()

        # Add labels and title
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("h")
        ax.legend()

        ax.set_xlim(17, 100)
        ax.set_yscale("log")

        fig.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        fig2.savefig(output_path.replace(".pdf", "_chi2.pdf"), bbox_inches="tight")
        if show:
            plt.figure(fig2.number)
            plt.show()
        plt.close(fig2)


def main():
    method_plots = MethodPlots2(
        id="0001",
        N_MAX=3,
        T=100,
        T0_REF=17,
        num_samples=int(1e3),
        # large_num_samples=int(7e6),
        large_num_samples=int(1e1),
        include_Mf=True,
        include_chif=True,
    )

    # method_plots.qnm_list += [(3,2,0,1)]
    # method_plots.spherical_modes += [(3,2)]
    # method_plots._initialize_results()

    method_plots.load_tuned_parameters()
    method_plots.compute_mf_chif()
    method_plots.get_t0_ref_fits()

    method_plots.ppc(output_path="outputs/ppc.pdf", show=False)
    # method_plots.linear_approximation(output_path="outputs/linear_approximation.pdf", show=False)
    # method_plots.plot_fundamental_kde(output_path="outputs/fundamental_kde.pdf", show=False)
    # method_plots.plot_overtone_kde(output_path="outputs/overtone_kde.pdf", show=False)
    # method_plots.plot_mass_spin_corner(output_path="outputs/mass_spin_corner.pdf", show=False)


if __name__ == "__main__":
    main()
